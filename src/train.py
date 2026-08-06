"""
train.py

Fine-tunes an mBART-50 many-to-many model on a custom parallel-text dataset.

Expects a JSONL file where each line has "english" and "persian" keys, e.g.:
    {"english": "God is merciful.", "persian": "خدا رحیم است."}

To use a different language pair, adjust --src-lang / --tgt-lang to any
mBART-50 language code (see the model card for the full list) and rename
the "english"/"persian" keys in your data (or pass --src-key / --tgt-key).

--base-model can point at any mBART-50-family checkpoint on the Hub (e.g.
another fine-tuned mBART-50 variant), but not at other architectures --
this script loads it via MBartForConditionalGeneration / MBart50TokenizerFast,
which are specific to mBART's tokenizer format and language-code system.

Usage:
    python train.py --data-file dataset.jsonl --output-dir mbart-fa-religious

    # Custom hyperparameters:
    python train.py \\
        --data-file dataset.jsonl \\
        --output-dir my-model \\
        --base-model facebook/mbart-large-50-many-to-many-mmt \\
        --src-lang en_XX --tgt-lang fa_IR \\
        --epochs 3 --learning-rate 2e-5 \\
        --batch-size 4 --grad-accum-steps 8 \\
        --warmup-ratio 0.1 --weight-decay 0.01 --num-beams 4

    # Push the result to the Hugging Face Hub when done:
    python train.py --data-file dataset.jsonl --output-dir my-model --push-to-hub my-username/my-model
"""

import argparse

from datasets import load_dataset
from transformers import (
    DataCollatorForSeq2Seq,
    MBart50TokenizerFast,
    MBartForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    # Data
    parser.add_argument("--data-file", required=True, help="Path to JSONL dataset.")
    parser.add_argument("--src-key", default="english", help="Source-language JSON key.")
    parser.add_argument("--tgt-key", default="persian", help="Target-language JSON key.")
    parser.add_argument("--test-size", type=float, default=0.1, help="Held-out fraction for eval.")
    parser.add_argument("--max-length", type=int, default=256, help="Max token length for src/tgt.")

    # Model / languages
    parser.add_argument(
        "--base-model",
        default="facebook/mbart-large-50-many-to-many-mmt",
        help="Base pretrained model to fine-tune. Must be an mBART-50 checkpoint "
        "(loaded via MBartForConditionalGeneration / MBart50TokenizerFast) -- "
        "e.g. another fine-tuned mBART-50 variant on the Hub. Not compatible "
        "with other architectures (NLLB, T5, MarianMT, etc.), which use "
        "different tokenizer classes and language-code conventions.",
    )
    parser.add_argument("--src-lang", default="en_XX", help="mBART-50 source language code.")
    parser.add_argument("--tgt-lang", default="fa_IR", help="mBART-50 target language code.")

    # Output
    parser.add_argument("--output-dir", required=True, help="Where to save checkpoints/final model.")
    parser.add_argument(
        "--push-to-hub",
        default=None,
        help="If set, push the final model to this Hugging Face Hub repo id "
        "(requires prior `huggingface-cli login` or notebook_login()).",
    )

    # Training hyperparameters
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Per-device train/eval batch size. Increase if your GPU has memory "
        "headroom -- combined with --grad-accum-steps this sets the effective "
        "batch size (default 4 x 8 = 32).",
    )
    parser.add_argument(
        "--grad-accum-steps",
        type=int,
        default=8,
        help="Gradient accumulation steps. Effective batch size = "
        "--batch-size x --grad-accum-steps.",
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.1,
        help="Fraction of total training steps used for linear LR warmup "
        "before ramping to --learning-rate. Helps stabilize early fine-tuning "
        "of a large pretrained model.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="AdamW weight decay, a standard regularizer to reduce overfitting.",
    )
    parser.add_argument(
        "--num-beams",
        type=int,
        default=4,
        help="Beam search width used during eval/generation (predict_with_generate). "
        "num_beams=1 is greedy decoding; 4-5 is a common default that noticeably "
        "improves translation quality over greedy at modest extra eval cost.",
    )
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.0,
        help="Label smoothing factor. Defaults to 0 -- with mBART-50's large "
        "(250k-token) vocabulary, label smoothing's second loss term sums "
        "over the entire vocabulary and can dominate the reported training "
        "loss, making it look far worse than the model's real performance "
        "(eval_loss, computed without smoothing, is unaffected). Set > 0 "
        "if you specifically want the regularization and are aware the "
        "training loss number will look inflated as a result.",
    )
    parser.add_argument(
        "--eval-save-strategy",
        choices=["epoch", "steps"],
        default="epoch",
        help="'epoch' (default) evaluates/saves once per epoch, so the last "
        "epoch's checkpoint always aligns exactly with the true end of "
        "training -- no remainder-steps problem. 'steps' evaluates/saves "
        "every --eval-save-steps steps instead; use this only if you want "
        "checkpoints more granular than once per epoch, and be aware any "
        "steps after the last save point are trained but never checkpointed.",
    )
    parser.add_argument(
        "--eval-save-steps",
        type=int,
        default=5000,
        help="Only used when --eval-save-strategy=steps. Pick a value that "
        "divides evenly into your expected total steps "
        "(train_examples / effective_batch_size * epochs).",
    )
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--no-fp16", dest="fp16", action="store_false")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # --- Load and reshape the dataset ---
    raw_dataset = load_dataset("json", data_files=args.data_file, split="train")

    def to_translation_pair(example):
        return {
            "translation": {
                "en": example[args.src_key],
                "fa": example[args.tgt_key],
            }
        }

    dataset = raw_dataset.map(to_translation_pair)
    dataset = dataset.remove_columns([args.src_key, args.tgt_key])
    split = dataset.train_test_split(test_size=args.test_size)
    train_dataset, eval_dataset = split["train"], split["test"]
    print(f"Train examples: {len(train_dataset)} | Eval examples: {len(eval_dataset)}")

    effective_batch_size = args.batch_size * args.grad_accum_steps
    steps_per_epoch = max(1, len(train_dataset) // effective_batch_size)
    total_steps = int(steps_per_epoch * args.epochs)
    print(
        f"Effective batch size: {effective_batch_size} | "
        f"Steps/epoch: {steps_per_epoch} | Estimated total steps: {total_steps}"
    )
    if args.eval_save_strategy == "steps" and args.eval_save_steps > total_steps:
        print(
            f"WARNING: --eval-save-steps ({args.eval_save_steps}) is larger than the "
            f"estimated total steps ({total_steps}). Evaluation will never fire during "
            f"training -- consider lowering it (e.g. total_steps // 3), or use "
            f"--eval-save-strategy epoch instead."
        )

    # --- Load base model and tokenizer ---
    tokenizer = MBart50TokenizerFast.from_pretrained(args.base_model)
    model = MBartForConditionalGeneration.from_pretrained(args.base_model)

    tokenizer.src_lang = args.src_lang
    tokenizer.tgt_lang = args.tgt_lang

    def preprocess(batch):
        src = [x["en"] for x in batch["translation"]]
        tgt = [x["fa"] for x in batch["translation"]]

        model_inputs = tokenizer(
            src, text_target=tgt, max_length=args.max_length, truncation=True
        )

        return model_inputs

    train_tokenized = train_dataset.map(
        preprocess, batched=True, remove_columns=train_dataset.column_names
    )
    eval_tokenized = eval_dataset.map(
        preprocess, batched=True, remove_columns=eval_dataset.column_names
    )

    # --- Training arguments ---
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        num_train_epochs=args.epochs,
        fp16=args.fp16,
        logging_steps=args.logging_steps,
        eval_strategy=args.eval_save_strategy,
        eval_steps=args.eval_save_steps if args.eval_save_strategy == "steps" else None,
        save_strategy=args.eval_save_strategy,
        save_steps=args.eval_save_steps if args.eval_save_strategy == "steps" else None,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        predict_with_generate=True,
        generation_num_beams=args.num_beams,
        remove_unused_columns=True,
        label_smoothing_factor=args.label_smoothing,
        push_to_hub=bool(args.push_to_hub),
        hub_model_id=args.push_to_hub if args.push_to_hub else None,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer, model=model, label_pad_token_id=-100
    )

    # --- Generation config fixes ---
    # mBART-50's pretrained checkpoint ships forced_bos_token_id in model.config,
    # which newer transformers versions reject at save time (generation params
    # belong in model.generation_config, not model.config). Clear the legacy
    # copy and set the correct values in the right place.
    #
    # Critically, decoder_start_token_id must stay as the eos token (matching
    # model.config.decoder_start_token_id, used during training's label-shifting) --
    # NOT the target language tag. Setting it to the language tag here breaks
    # inference: generation would skip the true start-of-sequence state the
    # decoder was actually trained on, misaligning every generated token.
    model.generation_config.decoder_start_token_id = tokenizer.eos_token_id
    model.generation_config.forced_bos_token_id = tokenizer.lang_code_to_id[args.tgt_lang]
    model.config.forced_bos_token_id = None

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=eval_tokenized,
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    # --- Train ---
    trainer.train()

    # --- Confirm final metrics ---
    # Because load_best_model_at_end=True, the trainer has already swapped in
    # whichever evaluated checkpoint had the lowest eval_loss -- this call
    # doesn't produce a "new" number, it just prints that checkpoint's metrics
    # clearly rather than requiring you to dig them out of the training logs.
    final_metrics = trainer.evaluate()
    print("Metrics for the best (saved) checkpoint:", final_metrics)

    # --- Save locally ---
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Model saved to: {args.output_dir}")

    # --- Optionally push to the Hub ---
    if args.push_to_hub:
        trainer.push_to_hub(args.push_to_hub)
        print(f"Model pushed to: https://huggingface.co/{args.push_to_hub}")


if __name__ == "__main__":
    main()

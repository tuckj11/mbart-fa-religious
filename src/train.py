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
        --batch-size 1 --grad-accum-steps 8

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
    parser.add_argument("--batch-size", type=int, default=1, help="Per-device train/eval batch size.")
    parser.add_argument("--grad-accum-steps", type=int, default=8)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--eval-save-steps", type=int, default=10000)
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

    # --- Load base model and tokenizer ---
    tokenizer = MBart50TokenizerFast.from_pretrained(args.base_model)
    model = MBartForConditionalGeneration.from_pretrained(args.base_model)

    tokenizer.src_lang = args.src_lang
    tokenizer.tgt_lang = args.tgt_lang

    def preprocess(batch):
        src = [x["en"] for x in batch["translation"]]
        tgt = [x["fa"] for x in batch["translation"]]

        model_inputs = tokenizer(
            src, max_length=args.max_length, truncation=True, padding=True
        )

        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                tgt, max_length=args.max_length, truncation=True, padding=True
            )["input_ids"]

        model_inputs["labels"] = labels
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
        num_train_epochs=args.epochs,
        fp16=args.fp16,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_save_steps,
        eval_strategy="steps",
        save_steps=args.eval_save_steps,
        save_strategy="steps",
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        predict_with_generate=True,
        remove_unused_columns=True,
        label_smoothing_factor=args.label_smoothing,
        push_to_hub=bool(args.push_to_hub),
        hub_model_id=args.push_to_hub if args.push_to_hub else None,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer, model=model, label_pad_token_id=tokenizer.pad_token_id
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=eval_tokenized,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # --- Train ---
    trainer.train()

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

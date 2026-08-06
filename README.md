# English → Persian Religious Text Translation (mBART-50 Fine-Tune)

A fine-tuned neural machine translation model for English-to-Persian (Farsi) translation of religious and doctrinal text, built end-to-end: custom web-scraped parallel corpus → strict alignment filtering → fine-tuning `facebook/mbart-large-50-many-to-many-mmt` → evaluation.

**Model on Hugging Face:** [mbart-fa-religious-final]([tuckj90/mbart-fa-religious-final](https://huggingface.co/tuckj90/mbart-fa-religious-final)) 

## Overview

Off-the-shelf multilingual MT models handle general-purpose text reasonably well, but often struggle with the specific register, vocabulary, and phrasing conventions of religious/doctrinal writing. This project builds a domain-adapted EN→FA translator by fine-tuning mBART-50 on a purpose-built parallel corpus of religious text.

## Data Pipeline

1. **Scraping** ([`src/scrape_data.py`](src/scrape_data.py)): Paired English/Persian pages were scraped from published religious texts, including General Conference talks (2014–2025), *Liahona* messages, and several doctrinal manuals.
2. **Strict alignment filtering:** Rather than naively pairing sentences, the pipeline:
   - Discards an entire document if the English and Persian paragraph counts don't match.
   - Within each paragraph, discards it unless the sentence counts align exactly on both sides.

   This trades dataset size for alignment quality — noisy sentence alignment is one of the most common failure modes in parallel-corpus MT projects, so bad pairs are dropped rather than kept.
3. **Result:** ~49,000 aligned sentence pairs, split 90/10 into train/test.

## Model & Training

| Setting | Value |
|---|---|
| Base model | `facebook/mbart-large-50-many-to-many-mmt` |
| Source → target | `en_XX` → `fa_IR` |
| Effective batch size | 32 (batch size 4 × 8 gradient accumulation steps) |
| Learning rate | 2e-5, with 10% linear warmup |
| Weight decay | 0.01 |
| Epochs | 3 |
| Precision | fp16 (mixed precision) |
| Eval/save strategy | Once per epoch (guarantees the final checkpoint reflects the true end of training) |
| Generation during eval | Beam search, 4 beams |
| Hardware | Google Colab, A100 GPU |

Trained with [`src/train.py`](src/train.py); the exact run used for the published model is recorded in [`notebooks/training_notebook.ipynb`](notebooks/training_notebook.ipynb).

## Results

Evaluated on the full held-out test split (~4,900 sentence pairs).

| Metric | Score |
|---|---|
| Validation loss | **1.171** |
| BLEU | **26.27** |
| Precisions (1-4 gram) | 58.1% / 32.7% / 19.9% / 12.6% |
| Brevity penalty | 1.0 (no length penalty — outputs aren't under-generating) |
| Length ratio | 1.046 |

For context, general-domain English→Persian MT systems commonly score in the 15–25 BLEU range; a domain-adapted score in the mid-20s, measured on the full test set (not a small sample), reflects a solid, well-supported result for a religious/doctrinal-register translator.

**Example:**
> *"God said, let there be light"* → **خدا گفت، بگذار نور باشد**

See [`examples/sample_translations.md`](examples/sample_translations.md) for more.

## Usage

```python
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

tokenizer = MBart50TokenizerFast.from_pretrained("your-username/mbart-fa-religious-final", src_lang="en_XX")
model = MBartForConditionalGeneration.from_pretrained("your-username/mbart-fa-religious-final")
model.eval()

def translate(text, num_beams=4):
    encoded = tokenizer(text, return_tensors="pt")
    generated_tokens = model.generate(
        **encoded,
        forced_bos_token_id=tokenizer.lang_code_to_id["fa_IR"],
        num_beams=num_beams,
    )
    return tokenizer.decode(generated_tokens[0], skip_special_tokens=True)

translate("And God said, Let there be light.")
```

Or use [`src/translate.py`](src/translate.py) directly (`python translate.py "some text"`).

To fine-tune on your own dataset, see [`src/train.py`](src/train.py) — it accepts any JSONL file of source/target pairs and is not tied to this project's specific corpus.

## Repository Structure

```
├── notebooks/    Training notebook — the documented record of the actual published run
├── src/          Reusable scripts: scrape_data.py, train.py, translate.py
├── examples/     Sample translations
└── requirements.txt
```

## Development Notes

A few non-obvious issues came up during development that are worth documenting for anyone extending this project:

- **Label smoothing inflates training loss with large vocabularies.** With mBART-50's ~250k-token vocabulary, `label_smoothing_factor > 0` sums a small per-token penalty across the *entire* vocabulary, which can dominate the reported training loss and make it look far worse than the model's real performance. `eval_loss` (computed without smoothing) is unaffected. This project trains with `label_smoothing_factor=0` for a training-loss number that's actually interpretable.
- **`decoder_start_token_id` must be the eos token, not the target-language tag**, even though `forced_bos_token_id` is the language tag. Setting `decoder_start_token_id` to the language tag breaks generation — it skips the true sequence-start state the decoder was trained on, misaligning every subsequent generated token and producing gibberish, even though the underlying trained weights are completely fine.
- **`eval_strategy`/`save_strategy="steps"` with a fixed step count can silently leave the true final training state unevaluated and unsaved** if total steps aren't an exact multiple of the chosen interval. `"epoch"` strategy avoids this — the last epoch's checkpoint always aligns exactly with the true end of training.

## Framework Versions

- Transformers 4.57.3
- PyTorch 2.9.0+cu126
- Datasets 4.0.0
- Tokenizers 0.22.1

## Notes

- Model weights are hosted on Hugging Face (not this repo) — see link above.
- The training corpus is not included here due to source licensing; the scraper script shows the collection methodology.


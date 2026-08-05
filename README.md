# English → Persian Religious Text Translation (mBART-50 Fine-Tune)

A fine-tuned neural machine translation model for English-to-Persian (Farsi) translation of religious and doctrinal text, built end-to-end: custom web-scraped parallel corpus → strict alignment filtering → fine-tuning `facebook/mbart-large-50-many-to-many-mmt` → evaluation.

**Model on Hugging Face:** [mbart-fa-religious](#) <!-- replace with your actual HF model URL -->

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
| Effective batch size | 8 (batch size 1 × 8 gradient accumulation steps) |
| Learning rate | 2e-5 |
| Epochs | 3 |
| Precision | fp16 (mixed precision) |
| Label smoothing | 0.1 |
| Hardware | Google Colab, A100 GPU |

Training and evaluation loss stayed close together throughout training (train: 1.75, val: 1.81 at the reported checkpoint), indicating the model generalized rather than memorized the training set.

## Results

| Metric | Score |
|---|---|
| BLEU | **35.55** |
| Precisions (1-4 gram) | 57.0% / 41.4% / 30.0% / 22.6% |
| Brevity penalty | 1.0 (no length penalty — outputs aren't under-generating) |

For context, general-domain English→Persian MT systems often score in the 15–25 BLEU range; a domain-adapted score in the mid-30s on in-domain text reflects the benefit of fine-tuning on carefully aligned, topic-matched data.

## Example Translations

See [`examples/sample_translations.md`](examples/sample_translations.md) for input/output pairs.

## Usage

```python
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

tokenizer = MBart50TokenizerFast.from_pretrained("your-username/mbart-fa-religious-final", src_lang="en_XX")
model = MBartForConditionalGeneration.from_pretrained("your-username/mbart-fa-religious-final")

def translate(text):
    encoded = tokenizer(text, return_tensors="pt")
    generated_tokens = model.generate(
        **encoded,
        forced_bos_token_id=tokenizer.lang_code_to_id["fa_IR"]
    )
    return tokenizer.decode(generated_tokens[0], skip_special_tokens=True)

translate("And God said, Let there be light.")
```

## Repository Structure

```
├── notebooks/    Training notebook (data prep → fine-tuning → evaluation)
├── src/          Standalone scraper and inference scripts
├── examples/     Sample translations
└── requirements.txt
```

## Framework Versions

- Transformers 4.57.3
- PyTorch 2.9.0+cu126
- Datasets 4.0.0
- Tokenizers 0.22.1

## Notes

- Model weights are hosted on Hugging Face (not this repo) — see link above.
- The training corpus is not included here due to source licensing; the scraper script shows the collection methodology.

"""
translate.py

Loads the fine-tuned mBART-50 English->Persian model and translates text.

Usage:
    python translate.py "And God said, Let there be light."

Or import translate() directly:
    from translate import translate
    translate("Some English text.")
"""

import sys

import torch
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

MODEL_NAME = "your-username/mbart-fa-religious-final"  # replace with your HF model path

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = MBart50TokenizerFast.from_pretrained(MODEL_NAME, src_lang="en_XX")
model = MBartForConditionalGeneration.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()

# Defensive fix: decoder_start_token_id must be the eos token (matching how
# the decoder was actually trained via label-shifting), not the target
# language tag. If it's ever set to the language tag instead -- e.g. from an
# older/manually-patched checkpoint -- generation silently misaligns every
# token and produces gibberish output, even though the underlying weights
# are fine. This override is a no-op if the checkpoint is already correct.
model.generation_config.decoder_start_token_id = tokenizer.eos_token_id


def translate(text: str, num_beams: int = 4) -> str:
    """Translate a single English string to Persian using beam search."""
    encoded = tokenizer(text, return_tensors="pt").to(DEVICE)

    generated_tokens = model.generate(
        **encoded,
        forced_bos_token_id=tokenizer.lang_code_to_id["fa_IR"],
        num_beams=num_beams,
    )

    return tokenizer.decode(generated_tokens[0], skip_special_tokens=True)


def translate_batch(texts: list[str]) -> list[str]:
    """Translate a list of English strings to Persian."""
    return [translate(t) for t in texts]


if __name__ == "__main__":
    if len(sys.argv) > 1:
        input_text = " ".join(sys.argv[1:])
    else:
        input_text = "And God said, Let there be light."

    print(translate(input_text))


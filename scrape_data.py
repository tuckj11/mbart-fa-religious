"""
scrape_data.py

Scrapes paired English/Persian pages and produces a strictly-aligned
parallel corpus of sentence pairs, saved as JSONL.

Alignment strategy:
    - An entire document is discarded if its English and Persian paragraph
      counts don't match.
    - Within an accepted document, each paragraph is discarded unless its
      English and Persian sentence counts match exactly.

This trades corpus size for alignment quality: rather than trying to
fix or approximate misaligned text, mismatches are simply dropped.

Usage:
    python scrape_data.py

Edit TALK_PAIRS below (or load them from a file) with your own list of
(english_url, persian_url) tuples before running.
"""

import json
import re

import requests
from bs4 import BeautifulSoup

# ======================================================
# *** Update these for your source site ***
CONTENT_SELECTOR = ".body-block"
OUTPUT_FILE = "dataset.jsonl"
PARAGRAPH_TAG = "p"
# ======================================================


def clean_text(text: str) -> str:
    """Normalize text and remove excessive whitespace/newlines."""
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_text_into_sentences(text: str, lang: str = "en") -> list[str]:
    """Segments text into sentences based on punctuation."""
    if not text:
        return []
    if lang == "fa":
        # Split on Persian/Arabic end-of-sentence punctuation (. ! ? ؛ ؟)
        sentences = re.split(r"(?<=[.?!؛؟])\s+", text)
    else:
        # Split on standard punctuation (. ! ?)
        sentences = re.split(r"(?<=[.?!])\s+", text)

    return [s.strip() for s in sentences if s.strip()]


def scrape_url_by_elements(
    url: str, selector: str, paragraph_tag: str = "p"
) -> list[str]:
    """
    Fetches text from a single URL and extracts the content as a list of
    paragraphs by finding all elements matching paragraph_tag within the
    main container matched by selector.
    """
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")

        main_element = soup.select_one(selector)
        if main_element:
            paragraph_elements = main_element.find_all(paragraph_tag)
            paragraphs = [clean_text(p.get_text()) for p in paragraph_elements]
            return [p for p in paragraphs if p]

    except Exception as e:
        print(f"❌ Error scraping {url}: {e}")

    return []


def scrape_and_process_pair_strict(
    index: int,
    en_url: str,
    fa_url: str,
    selector: str,
    output_file: str,
    paragraph_tag: str = "p",
) -> None:
    """Fetches, strictly aligns by paragraph and sentence count, and appends
    the resulting pairs to output_file as JSONL."""

    en_paragraphs = scrape_url_by_elements(en_url, selector, paragraph_tag)
    fa_paragraphs = scrape_url_by_elements(fa_url, selector, paragraph_tag)

    print(f"Document #{index}")
    print(f"  English paragraphs: {len(en_paragraphs)}")
    print(f"  Persian paragraphs: {len(fa_paragraphs)}")

    if not en_paragraphs or not fa_paragraphs:
        print(f"⚠️  Skipped (could not retrieve/parse): {en_url}")
        return

    if len(en_paragraphs) != len(fa_paragraphs):
        print(
            f"⚠️  Skipped (paragraph count mismatch: "
            f"EN {len(en_paragraphs)} vs FA {len(fa_paragraphs)}): {en_url}"
        )
        return

    saved_sentences = 0
    discarded_paragraphs = 0

    with open(output_file, "a", encoding="utf-8") as f_out:
        for en_para, fa_para in zip(en_paragraphs, fa_paragraphs):
            en_sentences = split_text_into_sentences(en_para, lang="en")
            fa_sentences = split_text_into_sentences(fa_para, lang="fa")

            if len(en_sentences) == len(fa_sentences) and len(en_sentences) > 0:
                for en, fa in zip(en_sentences, fa_sentences):
                    entry = {"english": en, "persian": fa}
                    f_out.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    saved_sentences += 1
            else:
                discarded_paragraphs += 1

    print(
        f"✅ Saved {saved_sentences} pairs. "
        f"Discarded {discarded_paragraphs} misaligned/empty paragraphs.\n"
    )


# Populate with your own list of (english_url, persian_url) pairs.
TALK_PAIRS: list[tuple[str, str]] = [
    # ("https://example.com/en/page", "https://example.com/fa/page"),
]


def main() -> None:
    # Clear the output file before starting a new run
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        pass

    for index, (en_url, fa_url) in enumerate(TALK_PAIRS):
        scrape_and_process_pair_strict(
            index, en_url, fa_url, CONTENT_SELECTOR, OUTPUT_FILE, PARAGRAPH_TAG
        )


if __name__ == "__main__":
    main()

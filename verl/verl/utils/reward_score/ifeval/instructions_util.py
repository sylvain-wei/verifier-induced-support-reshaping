import re
import string

import immutabledict
import nltk


LANGUAGE_CODES = immutabledict.immutabledict(
    {
        "english": "en",
        "en": "en",
        "french": "fr",
        "fr": "fr",
        "german": "de",
        "de": "de",
        "spanish": "es",
        "es": "es",
        "italian": "it",
        "it": "it",
        "portuguese": "pt",
        "pt": "pt",
        "dutch": "nl",
        "nl": "nl",
    }
)


def normalize_space(text: str) -> str:
    return " ".join(text.strip().split())


def strip_punctuation(word: str) -> str:
    return word.strip(string.punctuation)


def split_into_sentences(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    try:
        sents = nltk.sent_tokenize(text)
    except LookupError:
        sents = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sents if s.strip()]


def word_tokenize(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    try:
        tokens = nltk.word_tokenize(text)
    except LookupError:
        tokens = re.findall(r"\b\w+\b|[^\w\s]", text, re.UNICODE)
    return tokens


def split_by_markdown_divider(text: str) -> list[str]:
    parts = re.split(r"\n\s*(?:\*\s*\*\s*\*|\*{3})\s*\n|\s*(?:\*\s*\*\s*\*|\*{3})\s*", text.strip())
    return [p.strip() for p in parts if p.strip()]


def count_words(text: str) -> int:
    return len([t for t in word_tokenize(text) if re.match(r"\w+", t)])


def count_letter(text: str, letter: str) -> int:
    return text.count(letter)

# Copyright 2026 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Adapted and reduced for the verifier coverage released with this project.

import re
import string

import immutabledict

# We intentionally avoid requiring nltk corpora here: the IFEval constraint
# checks only need sentence/word splits that can be done with regex, and
# sentence tokenization quality is not critical for the boolean pass/fail.


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
    sents = re.split(r"(?<=[.!?])\s+", text)
    out: list[str] = []
    for s in sents:
        for part in s.split("\n"):
            part = part.strip()
            if part:
                out.append(part)
    return out


def word_tokenize(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    return re.findall(r"\b\w+\b|[^\w\s]", text, re.UNICODE)


def split_by_markdown_divider(text: str) -> list[str]:
    parts = re.split(r"\n\s*(?:\*\s*\*\s*\*|\*{3})\s*\n|\s*(?:\*\s*\*\s*\*|\*{3})\s*", text.strip())
    return [p.strip() for p in parts if p.strip()]


def count_words(text: str) -> int:
    return len([t for t in word_tokenize(text) if re.match(r"\w+", t)])


def count_letter(text: str, letter: str) -> int:
    return text.count(letter)

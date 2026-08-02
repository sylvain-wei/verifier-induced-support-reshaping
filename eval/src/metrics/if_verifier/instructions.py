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

import json
import re
from abc import ABC, abstractmethod

import langdetect

from . import instructions_util as iu


def _extract_last_word(text: str) -> str:
    words = [w for w in re.findall(r"\b\w+\b", text)]
    return words[-1].lower() if words else ""


def _word_count_for_keyword(text: str, keyword: str) -> int:
    words = [w.lower() for w in re.findall(r"\b\w+\b", text.lower())]
    return sum(1 for w in words if w == keyword.lower())


def _check_relation(actual: int, target: int, relation: str | None) -> bool:
    relation = (relation or "at least").strip().lower()
    if relation in {"at least", ">=", "more than", "not less than"}:
        return actual >= target
    if relation in {"less than", "<"}:
        return actual < target
    if relation in {"at most", "<=", "no more than"}:
        return actual <= target
    if relation in {"around", "approximately"}:
        tolerance = max(1, int(target * 0.1))
        return abs(actual - target) <= tolerance
    return actual == target


class Instruction(ABC):
    def __init__(self, instruction_id: str):
        self.instruction_id = instruction_id
        self.args = {}

    def build_description(self, **kwargs):
        self.args = kwargs
        return ""

    @abstractmethod
    def check_following(self, value: str) -> bool:
        raise NotImplementedError


class SimpleInstruction(Instruction):
    def check_following(self, value: str) -> bool:
        return check_constraint(self.instruction_id, value, self.args)


def check_constraint(instruction_id: str, value: str, args: dict) -> bool:
    text = value.strip()
    if not text:
        return False

    if instruction_id == "keywords:existence":
        keywords = args.get("keywords") or ([args.get("keyword")] if args.get("keyword") else [])
        text_l = text.lower()
        return all(k and k.lower() in text_l for k in keywords)

    if instruction_id == "keywords:frequency":
        keyword = args.get("keyword")
        frequency = args.get("frequency")
        relation = args.get("relation")
        if not keyword or frequency is None:
            return False
        return _check_relation(_word_count_for_keyword(text, keyword), int(frequency), relation)

    if instruction_id == "keywords:forbidden_words":
        forbidden_words = args.get("forbidden_words") or ([args.get("keyword")] if args.get("keyword") else [])
        text_l = text.lower()
        return all(w.lower() not in text_l for w in forbidden_words if w)

    if instruction_id == "keywords:letter_frequency":
        letter = args.get("letter")
        target = args.get("let_frequency") if args.get("let_frequency") is not None else args.get("frequency")
        relation = args.get("let_relation") or args.get("relation")
        if not letter or target is None:
            return False
        return _check_relation(iu.count_letter(text, letter), int(target), relation)

    if instruction_id == "language:response_language":
        lang = (args.get("language") or "en").lower()
        lang = iu.LANGUAGE_CODES.get(lang, lang)
        try:
            detected = langdetect.detect(text)
        except Exception:
            return False
        return detected == lang

    if instruction_id == "length_constraints:number_sentences":
        target = args.get("num_sentences")
        relation = args.get("relation")
        if target is None:
            return False
        return _check_relation(len(iu.split_into_sentences(text)), int(target), relation)

    if instruction_id == "length_constraints:number_paragraphs":
        target = args.get("num_paragraphs")
        if target is None:
            return False
        return len(iu.split_by_markdown_divider(text)) == int(target)

    if instruction_id == "length_constraints:number_words":
        target = args.get("num_words")
        relation = args.get("relation")
        if target is None:
            return False
        return _check_relation(iu.count_words(text), int(target), relation)

    if instruction_id == "length_constraints:nth_paragraph_first_word":
        nth = args.get("nth_paragraph")
        first_word = args.get("first_word")
        if nth is None or not first_word:
            return False
        paragraphs = iu.split_by_markdown_divider(text)
        idx = int(nth) - 1
        if idx < 0 or idx >= len(paragraphs):
            return False
        tokens = [iu.strip_punctuation(t).lower() for t in iu.word_tokenize(paragraphs[idx]) if iu.strip_punctuation(t)]
        return bool(tokens) and tokens[0] == first_word.lower()

    if instruction_id == "detectable_content:number_placeholders":
        target = args.get("num_placeholders")
        if target is None:
            return False
        return len(re.findall(r"\[[^\[\]]+\]", text)) >= int(target)

    if instruction_id == "detectable_content:postscript":
        marker = args.get("postscript_marker") or "P.S."
        return marker in text

    if instruction_id == "detectable_format:number_bullet_lists":
        target = args.get("num_bullets")
        if target is None:
            return False
        bullets = [line for line in text.splitlines() if re.match(r"^\s*[-*]\s+\S+", line)]
        return len(bullets) == int(target)

    if instruction_id == "detectable_format:constrained_response":
        options = {"my answer is yes.", "my answer is no.", "my answer is maybe."}
        return text.strip().lower() in options

    if instruction_id == "detectable_format:number_highlighted_sections":
        target = args.get("num_highlights")
        if target is None:
            return False
        highlighted = re.findall(r"\*[^*\n][^*\n]*\*", text)
        return len(highlighted) >= int(target)

    if instruction_id == "detectable_format:multiple_sections":
        target = args.get("num_sections")
        splitter = args.get("section_spliter") or "Section"
        if target is None:
            return False
        return len(re.findall(re.escape(splitter), text, flags=re.IGNORECASE)) >= int(target)

    if instruction_id == "detectable_format:json_format":
        try:
            json.loads(text)
            return True
        except Exception:
            return False

    if instruction_id == "detectable_format:title":
        return bool(re.search(r"<<.+?>>", text, re.DOTALL))

    if instruction_id == "combination:two_responses":
        if text.count("******") != 1:
            return False
        a, b = [x.strip() for x in text.split("******")]
        return bool(a) and bool(b) and a != b

    if instruction_id == "combination:repeat_prompt":
        prompt_to_repeat = args.get("prompt_to_repeat")
        if not prompt_to_repeat:
            return False
        return text.startswith(prompt_to_repeat)

    if instruction_id == "startend:end_checker":
        end_phrase = args.get("end_phrase")
        return bool(end_phrase) and text.endswith(end_phrase)

    if instruction_id == "change_case:capital_word_frequency":
        target = args.get("capital_frequency")
        relation = args.get("capital_relation")
        if target is None:
            return False
        caps = re.findall(r"\b[A-Z]+\b", text)
        return _check_relation(len(caps), int(target), relation)

    if instruction_id == "change_case:english_capital":
        letters = [c for c in text if c.isalpha()]
        return bool(letters) and all(c.isupper() for c in letters)

    if instruction_id == "change_case:english_lowercase":
        letters = [c for c in text if c.isalpha()]
        return bool(letters) and all(c.islower() for c in letters)

    if instruction_id == "punctuation:no_comma":
        return "," not in text

    if instruction_id == "startend:quotation":
        return text.startswith('"') and text.endswith('"')

    if instruction_id == "copy:repeat_phrase":
        phrase = args.get("phrase") or args.get("prompt_to_repeat")
        count = args.get("N") or args.get("n") or 2
        if not phrase:
            return False
        return text.count(phrase) >= int(count)

    if instruction_id in {"copy:copy", "copy:copying_simple"}:
        prompt_to_repeat = args.get("prompt_to_repeat")
        return bool(prompt_to_repeat) and prompt_to_repeat in text

    if instruction_id == "new:copy_span_idx":
        span = args.get("span") or args.get("copy_span")
        if span:
            return str(span) in text
        source = args.get("source_text")
        sidx = args.get("start")
        eidx = args.get("end")
        if source is None or sidx is None or eidx is None:
            return False
        return source[int(sidx) : int(eidx)] in text

    if instruction_id == "detectable_format:sentence_hyphens":
        compact = text.replace(" ", "")
        return "-" in compact and re.search(r"[A-Za-z]-[A-Za-z]", compact) is not None

    if instruction_id == "keywords:no_adjacent_consecutive":
        keyword = args.get("keyword")
        if not keyword:
            return True
        lower = text.lower()
        return (keyword.lower() * 2) not in lower

    if instruction_id == "detectable_format:square_brackets":
        return bool(re.search(r"\[[^\[\]]+\]", text))

    if instruction_id == "keywords:word_once":
        keyword = args.get("keyword")
        return bool(keyword) and _word_count_for_keyword(text, keyword) == 1

    if instruction_id == "keywords:word_count_different_numbers":
        keyword = args.get("keyword")
        freq = args.get("frequency")
        if keyword and freq is not None:
            return _word_count_for_keyword(text, keyword) == int(freq)
        keyword_map = args.get("keyword_count_map")
        if isinstance(keyword_map, dict):
            return all(_word_count_for_keyword(text, k) == int(v) for k, v in keyword_map.items())
        return False

    if instruction_id == "keywords:exclude_word_harder":
        keyword = args.get("keyword")
        return bool(keyword) and keyword.lower() not in text.lower()

    if instruction_id in {"paragraphs:paragraphs", "paragraphs:paragraphs2"}:
        target = args.get("num_paragraphs")
        if target is None:
            target = args.get("N")
        if target is None:
            return False
        paras = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
        return len(paras) == int(target)

    if instruction_id == "first_word:first_word_sent":
        first_word = args.get("first_word")
        if not first_word:
            return False
        sents = iu.split_into_sentences(text)
        return bool(sents) and all(
            (lambda toks: bool(toks) and toks[0].lower() == first_word.lower())(
                [iu.strip_punctuation(t) for t in iu.word_tokenize(s) if iu.strip_punctuation(t)]
            )
            for s in sents
        )

    if instruction_id == "first_word:first_word_answer":
        first_word = args.get("first_word")
        tokens = [iu.strip_punctuation(t) for t in iu.word_tokenize(text) if iu.strip_punctuation(t)]
        return bool(first_word) and bool(tokens) and tokens[0].lower() == first_word.lower()

    if instruction_id == "last_word:last_word_sent":
        last_word = args.get("last_word")
        if not last_word:
            return False
        sents = iu.split_into_sentences(text)
        if not sents:
            return False
        for sent in sents:
            words = [iu.strip_punctuation(t) for t in iu.word_tokenize(sent) if iu.strip_punctuation(t)]
            if not words or words[-1].lower() != last_word.lower():
                return False
        return True

    if instruction_id == "last_word:last_word_answer":
        last_word = args.get("last_word")
        return bool(last_word) and _extract_last_word(text) == last_word.lower()

    if instruction_id == "detectable_format:bigram_wrapping":
        bigram = args.get("bigram")
        if not bigram:
            return bool(re.search(r"\[[^\]]+\s+[^\]]+\]", text))
        return f"[{bigram}]" in text or f"({bigram})" in text

    if instruction_id == "copy:copying_multiple":
        prompt_to_repeat = args.get("prompt_to_repeat")
        n = args.get("N") or args.get("n") or 2
        if not prompt_to_repeat:
            return False
        return text.count(prompt_to_repeat) >= int(n)

    if instruction_id == "punctuation:punctuation_dot":
        sents = iu.split_into_sentences(text)
        return bool(sents) and all(sent.rstrip().endswith(".") for sent in sents)

    if instruction_id == "punctuation:punctuation_exclamation":
        sents = iu.split_into_sentences(text)
        return bool(sents) and all(sent.rstrip().endswith("!") for sent in sents)

    if instruction_id == "count:lowercase_counting":
        target = args.get("N") or args.get("num_lowercase")
        relation = args.get("relation")
        if target is None:
            return False
        actual = sum(1 for c in text if c.islower())
        return _check_relation(actual, int(target), relation)

    if instruction_id in {"letters:letter_counting", "letters:letter_counting2"}:
        letter = args.get("letter")
        target = args.get("N") or args.get("let_frequency") or args.get("frequency")
        relation = args.get("relation") or args.get("let_relation")
        if not letter or target is None:
            return False
        return _check_relation(iu.count_letter(text, letter), int(target), relation)

    if instruction_id == "count:counting_composition":
        target = args.get("N") or args.get("count")
        relation = args.get("relation")
        if target is None:
            return False
        return _check_relation(iu.count_words(text), int(target), relation)

    if instruction_id == "count:count_unique":
        target = args.get("N") or args.get("count")
        relation = args.get("relation")
        if target is None:
            return False
        words = [w.lower() for w in re.findall(r"\b\w+\b", text)]
        return _check_relation(len(set(words)), int(target), relation)

    if instruction_id == "count:count_increment_word":
        word = args.get("keyword") or args.get("word")
        target = args.get("N") or args.get("count")
        if not word or target is None:
            return False
        return _word_count_for_keyword(text, word) >= int(target)

    if instruction_id == "keywords:palindrome":
        cleaned = re.sub(r"[^a-z0-9]", "", text.lower())
        return len(cleaned) > 1 and cleaned == cleaned[::-1]

    if instruction_id == "keywords:keyword_specific_position":
        keyword = args.get("keyword")
        n = args.get("n")
        m = args.get("m")
        if not keyword or n is None or m is None:
            return False
        sents = iu.split_into_sentences(text)
        nidx = int(n) - 1
        midx = int(m) - 1
        if nidx < 0 or nidx >= len(sents):
            return False
        words = [iu.strip_punctuation(w) for w in iu.word_tokenize(sents[nidx]) if iu.strip_punctuation(w)]
        if midx < 0 or midx >= len(words):
            return False
        return words[midx] == keyword

    if instruction_id == "keywords:start_end":
        words = [w.lower() for w in re.findall(r"\b\w+\b", text)]
        return len(words) >= 2 and words[0] == words[-1]

    return False

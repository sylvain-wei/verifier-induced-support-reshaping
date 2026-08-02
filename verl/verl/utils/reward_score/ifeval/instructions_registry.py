from . import instructions

_PARAGRAPH = "paragraphs:"
_KEYWORD = "keywords:"
_LETTER = "letters:"
_LANGUAGE = "language:"
_LENGTH = "length_constraints:"
_CONTENT = "detectable_content:"
_FORMAT = "detectable_format:"
_COMBINATION = "combination:"
_STARTEND = "startend:"
_CHANGE_CASES = "change_case:"
_PUNCTUATION = "punctuation:"
_NEW = "new:"
_COPY = "copy:"
_FIRSTWORD = "first_word:"
_LASTWORD = "last_word:"
_COUNT = "count:"

_S = instructions.SimpleInstruction

INSTRUCTION_DICT = {
    _KEYWORD + "existence": _S,
    _KEYWORD + "frequency": _S,
    _KEYWORD + "forbidden_words": _S,
    _KEYWORD + "letter_frequency": _S,
    _LANGUAGE + "response_language": _S,
    _LENGTH + "number_sentences": _S,
    _LENGTH + "number_paragraphs": _S,
    _LENGTH + "number_words": _S,
    _LENGTH + "nth_paragraph_first_word": _S,
    _CONTENT + "number_placeholders": _S,
    _CONTENT + "postscript": _S,
    _FORMAT + "number_bullet_lists": _S,
    _FORMAT + "constrained_response": _S,
    _FORMAT + "number_highlighted_sections": _S,
    _FORMAT + "multiple_sections": _S,
    _FORMAT + "json_format": _S,
    _FORMAT + "title": _S,
    _COMBINATION + "two_responses": _S,
    _COMBINATION + "repeat_prompt": _S,
    _STARTEND + "end_checker": _S,
    _CHANGE_CASES + "capital_word_frequency": _S,
    _CHANGE_CASES + "english_capital": _S,
    _CHANGE_CASES + "english_lowercase": _S,
    _PUNCTUATION + "no_comma": _S,
    _STARTEND + "quotation": _S,
    _COPY + "repeat_phrase": _S,
    _COPY + "copy": _S,
    _NEW + "copy_span_idx": _S,
    _FORMAT + "sentence_hyphens": _S,
    _KEYWORD + "no_adjacent_consecutive": _S,
    _FORMAT + "square_brackets": _S,
    _KEYWORD + "word_once": _S,
    _KEYWORD + "word_count_different_numbers": _S,
    _KEYWORD + "exclude_word_harder": _S,
    _PARAGRAPH + "paragraphs": _S,
    _PARAGRAPH + "paragraphs2": _S,
    _FIRSTWORD + "first_word_sent": _S,
    _FIRSTWORD + "first_word_answer": _S,
    _LASTWORD + "last_word_sent": _S,
    _LASTWORD + "last_word_answer": _S,
    _FORMAT + "bigram_wrapping": _S,
    _COPY + "copying_simple": _S,
    _COPY + "copying_multiple": _S,
    _PUNCTUATION + "punctuation_dot": _S,
    _PUNCTUATION + "punctuation_exclamation": _S,
    _COUNT + "lowercase_counting": _S,
    _LETTER + "letter_counting": _S,
    _LETTER + "letter_counting2": _S,
    _COUNT + "counting_composition": _S,
    _COUNT + "count_unique": _S,
    _COUNT + "count_increment_word": _S,
    _KEYWORD + "palindrome": _S,
    _KEYWORD + "keyword_specific_position": _S,
    _KEYWORD + "start_end": _S,
}

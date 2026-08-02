import os
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..")))

from src.metrics.if_verifier.verifier import verify_ifeval, remove_thinking_section
from src.metrics.if_verifier.ifbench_supported_ids import (
    FORMAT_ONLY_SUPPORTED_IDS, is_format_only_example,
)


class TestIFVerifier(unittest.TestCase):
    def test_punctuation_no_comma_pass(self):
        r = verify_ifeval("This has no commas here.", ["punctuation:no_comma"], [{}])
        self.assertTrue(r.strict_prompt_pass)

    def test_punctuation_no_comma_fail(self):
        r = verify_ifeval("Hello, world.", ["punctuation:no_comma"], [{}])
        self.assertFalse(r.strict_prompt_pass)
        self.assertEqual(r.num_satisfied, 0)

    def test_number_highlighted_sections(self):
        txt = "*hi one* *hi two* *hi three*"
        r = verify_ifeval(txt, ["detectable_format:number_highlighted_sections"], [{"num_highlights": 3}])
        self.assertTrue(r.strict_prompt_pass)

    def test_number_words_at_least(self):
        r = verify_ifeval("one two three four", ["length_constraints:number_words"], [{"num_words": 3, "relation": "at least"}])
        self.assertTrue(r.strict_prompt_pass)
        r2 = verify_ifeval("one two", ["length_constraints:number_words"], [{"num_words": 3, "relation": "at least"}])
        self.assertFalse(r2.strict_prompt_pass)

    def test_unsupported_id(self):
        r = verify_ifeval("x", ["totally:fake_id"], [{}])
        self.assertEqual(r.num_supported, 0)
        self.assertFalse(r.strict_prompt_pass)

    def test_thinking_section_stripped(self):
        t = "<think>secret reasoning</think>final text here"
        stripped = remove_thinking_section(t)
        self.assertEqual(stripped, "final text here")

    def test_multi_constraint(self):
        r = verify_ifeval(
            "no commas here and enough words to pass the limit here we go",
            ["punctuation:no_comma", "length_constraints:number_words"],
            [{}, {"num_words": 5, "relation": "at least"}],
        )
        self.assertEqual(r.num_satisfied, 2)
        self.assertTrue(r.strict_prompt_pass)

    def test_format_only_filter(self):
        self.assertTrue(is_format_only_example(["punctuation:no_comma", "length_constraints:number_words"]))
        self.assertFalse(is_format_only_example(["punctuation:no_comma", "totally:fake"]))
        self.assertGreater(len(FORMAT_ONLY_SUPPORTED_IDS), 40)


if __name__ == "__main__":
    unittest.main()

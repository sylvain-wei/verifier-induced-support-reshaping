import os
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..")))

from src.metrics.math_metrics import (
    extract_final_answer, is_equiv, score_response, aggregate_over_samples,
    normalize_numeric,
)


class TestExtract(unittest.TestCase):
    def test_boxed(self):
        v, m = extract_final_answer("...therefore \\boxed{42}.")
        self.assertEqual(v, "42")
        self.assertEqual(m, "boxed")

    def test_boxed_with_nested_latex(self):
        v, m = extract_final_answer("final is \\boxed{\\frac{3}{7}}")
        self.assertEqual(v.replace(" ", ""), "\\frac{3}{7}")
        self.assertEqual(m, "boxed")

    def test_answer_line(self):
        t = "Some reasoning.\nAnswer: 540"
        v, m = extract_final_answer(t)
        self.assertEqual(v, "540")
        self.assertEqual(m, "answer_line")

    def test_fallback(self):
        v, m = extract_final_answer("... blah blah 123")
        self.assertEqual(v, "123")
        self.assertEqual(m, "last_number")

    def test_none(self):
        v, m = extract_final_answer("")
        self.assertEqual(m, "none")


class TestEquiv(unittest.TestCase):
    def test_int(self):
        self.assertTrue(is_equiv("540", "540"))
        self.assertTrue(is_equiv("540", "540", answer_type="integer_0_999"))
        self.assertFalse(is_equiv("541", "540"))

    def test_latex_frac_exact(self):
        self.assertTrue(is_equiv("\\frac{1}{2}", "\\frac{1}{2}"))

    def test_float(self):
        self.assertTrue(is_equiv("0.5", "1/2"))


class TestNumericGSM8KCorrectness(unittest.TestCase):
    """GSM8K models frequently emit numeric answers with currency, commas,
    or units. These must all be scored as correct against the plain-number gold."""

    def test_boxed_dollar(self):
        self.assertTrue(is_equiv("\\boxed{\\$18}", "18", answer_type="numeric"))
        self.assertTrue(is_equiv("\\boxed{$18}", "18", answer_type="numeric"))

    def test_boxed_comma(self):
        self.assertTrue(is_equiv("\\boxed{70,000}", "70000", answer_type="numeric"))
        self.assertTrue(is_equiv("\\boxed{1,234,567}", "1234567", answer_type="numeric"))

    def test_boxed_units(self):
        self.assertTrue(is_equiv("\\boxed{18 dollars}", "18", answer_type="numeric"))
        self.assertTrue(is_equiv("\\boxed{25%}", "25", answer_type="numeric"))

    def test_boxed_combined_noise(self):
        self.assertTrue(is_equiv("\\boxed{\\$70,000}", "70000", answer_type="numeric"))

    def test_normalize_numeric_direct(self):
        self.assertEqual(normalize_numeric("\\boxed{$70,000}"), "70000")
        self.assertEqual(normalize_numeric("$18"), "18")
        self.assertEqual(normalize_numeric("18 dollars"), "18")
        self.assertEqual(normalize_numeric("18%"), "18")

    def test_last_latex_does_not_eat_currency(self):
        # This was a real bug: "$2 each = $18" triggered last_latex matching
        # the literal text between the $-signs, losing the 18.
        text = "She has 9 eggs at $2 each = $18"
        ext, method = extract_final_answer(text)
        # Should fall through to last_number and yield 18.
        self.assertEqual(ext, "18", msg=f"got ext={ext!r} method={method!r}")

    def test_equality_with_gold_formats(self):
        # Full round-trip: response with common model noise vs plain gold.
        samples = [
            ("... so she makes \\boxed{18}.", "18"),
            ("... \\boxed{\\$18}", "18"),
            ("... \\boxed{70,000}", "70000"),
            ("... \\boxed{18 dollars}", "18"),
            ("... so the answer is 18.", "18"),
            ("Working through: 16-3-4=9, 9*2=18", "18"),
        ]
        for resp, gold in samples:
            ext, method = extract_final_answer(resp)
            ok = is_equiv(ext, gold, answer_type="numeric")
            self.assertTrue(ok, msg=f"resp={resp!r} ext={ext!r} method={method!r} gold={gold!r}")


class TestNumericGSM8KCorrectness(unittest.TestCase):
    """GSM8K models frequently emit numeric answers with currency, commas,
    or units. These must all be scored as correct against the plain-number gold."""

    def test_boxed_dollar(self):
        self.assertTrue(is_equiv("\\boxed{\\$18}", "18", answer_type="numeric"))
        self.assertTrue(is_equiv("\\boxed{$18}", "18", answer_type="numeric"))

    def test_boxed_comma(self):
        self.assertTrue(is_equiv("\\boxed{70,000}", "70000", answer_type="numeric"))
        self.assertTrue(is_equiv("\\boxed{1,234,567}", "1234567", answer_type="numeric"))

    def test_boxed_units(self):
        self.assertTrue(is_equiv("\\boxed{18 dollars}", "18", answer_type="numeric"))
        self.assertTrue(is_equiv("\\boxed{25%}", "25", answer_type="numeric"))

    def test_boxed_combined_noise(self):
        self.assertTrue(is_equiv("\\boxed{\\$70,000}", "70000", answer_type="numeric"))

    def test_normalize_numeric_direct(self):
        self.assertEqual(normalize_numeric("\\boxed{$70,000}"), "70000")
        self.assertEqual(normalize_numeric("$18"), "18")
        self.assertEqual(normalize_numeric("18 dollars"), "18")
        self.assertEqual(normalize_numeric("18%"), "18")

    def test_last_latex_does_not_eat_currency(self):
        # This was a real bug: "$2 each = $18" triggered last_latex matching
        # the literal text between the $-signs, losing the 18.
        text = "She has 9 eggs at $2 each = $18"
        ext, method = extract_final_answer(text)
        # Should fall through to last_number and yield 18.
        self.assertEqual(ext, "18", msg=f"got ext={ext!r} method={method!r}")

    def test_equality_with_gold_formats(self):
        # Full round-trip: response with common model noise vs plain gold.
        samples = [
            ("... so she makes \\boxed{18}.", "18"),
            ("... \\boxed{\\$18}", "18"),
            ("... \\boxed{70,000}", "70000"),
            ("... \\boxed{18 dollars}", "18"),
            ("... so the answer is 18.", "18"),
            ("Working through: 16-3-4=9, 9*2=18", "18"),
        ]
        for resp, gold in samples:
            ext, method = extract_final_answer(resp)
            ok = is_equiv(ext, gold, answer_type="numeric")
            self.assertTrue(ok, msg=f"resp={resp!r} ext={ext!r} method={method!r} gold={gold!r}")


class TestAggregate(unittest.TestCase):
    def test_pass_at_k(self):
        samples = [
            {"correct": 1, "normalized_extracted": "42", "normalized_gold": "42"},
            {"correct": 0, "normalized_extracted": "40", "normalized_gold": "42"},
            {"correct": 1, "normalized_extracted": "42", "normalized_gold": "42"},
        ]
        a = aggregate_over_samples(samples)
        self.assertAlmostEqual(a["pass@1"], 2 / 3)
        self.assertAlmostEqual(a["best@k"], 1.0)
        self.assertAlmostEqual(a["maj@k"], 1.0)
        self.assertEqual(a["num_unique_answers"], 2)


if __name__ == "__main__":
    unittest.main()

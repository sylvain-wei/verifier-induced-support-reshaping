"""Run with: python -m pytest eval/tests -q
Or standalone: python eval/tests/test_text_metrics.py
"""
import os
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..")))

from src.metrics.text_metrics import compute_text_metrics


class TestTextMetrics(unittest.TestCase):
    def test_empty(self):
        m = compute_text_metrics("")
        self.assertEqual(m["response_length_chars"], 0)
        self.assertEqual(m["step_count"], 0)
        self.assertEqual(m["equation_count"], 0)
        self.assertIsNone(m["answer_position_ratio"])

    def test_steps_and_equations(self):
        t = ("Step 1: We compute $x=5$.\n"
             "Step 2: Therefore $y=x+1=6$.\n"
             "Let's verify: substitute $x=5$ back: $5+1=6$. Thus \\boxed{6}.")
        m = compute_text_metrics(t)
        self.assertGreater(m["step_count"], 0)
        self.assertGreater(m["equation_count"], 0)
        self.assertGreater(m["verification_marker_count"], 0)
        self.assertLessEqual(m["answer_position_ratio"], 1.0)
        self.assertGreater(m["answer_position_ratio"], 0.5)  # boxed is near end

    def test_prefix_and_first_sentence(self):
        t = "Hello world. Second sentence here."
        m = compute_text_metrics(t)
        self.assertEqual(m["first_sentence"], "Hello world.")
        self.assertTrue(m["prefix_8_tokens"].startswith("Hello world."))


if __name__ == "__main__":
    unittest.main()

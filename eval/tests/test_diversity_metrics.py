import os
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..")))

from src.metrics.diversity_metrics import distinct_n, self_bleu, compute_diversity


class TestDiversity(unittest.TestCase):
    def test_distinct(self):
        responses = ["a b c", "a b c", "a b c"]
        self.assertAlmostEqual(distinct_n(responses, 1), 3 / 9)

    def test_distinct_varied(self):
        responses = ["a b c", "d e f", "g h i"]
        self.assertAlmostEqual(distinct_n(responses, 1), 1.0)

    def test_self_bleu_identical_high(self):
        r = ["this is a test", "this is a test", "this is a test"]
        s = self_bleu(r)
        self.assertGreater(s, 0.5)  # similar -> high BLEU

    def test_self_bleu_different(self):
        r = ["the cat sat", "xylophones chortle wildly", "green apples bounce"]
        s = self_bleu(r)
        self.assertLess(s, 0.5)

    def test_full(self):
        r = ["the a b c", "the a b c d", "the a b c d e"]
        out = compute_diversity(r, [4, 5, 6])
        self.assertIn("distinct_2", out)
        self.assertIn("self_bleu_2", out)
        self.assertIn("response_length_mean", out)


if __name__ == "__main__":
    unittest.main()

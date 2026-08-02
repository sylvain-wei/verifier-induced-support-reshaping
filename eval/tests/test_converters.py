"""Smoke check: dataset converters can load from real local files and
produce unified records that satisfy the schema."""
import os
import sys
import unittest

import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(_HERE, ".."))
sys.path.insert(0, REPO)

from src.data import converters as C  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402


def load_parquet(path):
    return pq.read_table(path).to_pylist()


class TestConverters(unittest.TestCase):
    def setUp(self):
        with open(os.path.join(REPO, "configs/prompts.yaml")) as f:
            self.prompts_cfg = yaml.safe_load(f)
        with open(os.path.join(REPO, "configs/datasets.yaml")) as f:
            self.d_cfg = yaml.safe_load(f)

    def test_math500(self):
        p = os.environ.get("PROJECT_ROOT", ".") + "/data/math500/test.parquet"
        if not os.path.exists(p):
            self.skipTest("math500 parquet missing")
        recs = load_parquet(p)[:3]
        out = C.math500_from_parquet(recs, self.d_cfg["benchmarks"]["math500"], self.prompts_cfg)
        self.assertEqual(len(out), 3)
        self.assertIn("prompt", out[0])
        self.assertIn("boxed", out[0]["prompt"].lower() or "")

    def test_aime24(self):
        p = os.environ.get("PROJECT_ROOT", ".") + "/data/aime24/aime24_opd_val.parquet"
        if not os.path.exists(p):
            self.skipTest("aime24 parquet missing")
        recs = load_parquet(p)[:3]
        out = C.aime_from_parquet(recs, self.d_cfg["benchmarks"]["aime24"], self.prompts_cfg)
        self.assertEqual(len(out), 3)
        self.assertIn("Answer:", out[0]["prompt"])
        self.assertTrue(out[0]["id"].startswith("aime24_"))
        # RQ1 design principle: DO NOT prescribe reasoning style in prompt.
        # If any of these phrases appear in the final prompt, the default
        # behavioral pattern observation is contaminated.
        self.assertNotIn("step by step", out[0]["prompt"].lower())
        self.assertNotIn("think step", out[0]["prompt"].lower())

    def test_math500_no_cot_instruction(self):
        p = os.environ.get("PROJECT_ROOT", ".") + "/data/math500/test.parquet"
        if not os.path.exists(p):
            self.skipTest("math500 parquet missing")
        recs = load_parquet(p)[:3]
        out = C.math500_from_parquet(recs, self.d_cfg["benchmarks"]["math500"], self.prompts_cfg)
        for rec in out:
            self.assertNotIn("step by step", rec["prompt"].lower())
            self.assertNotIn("think step", rec["prompt"].lower())
        # RQ1 design principle: DO NOT prescribe reasoning style in prompt.
        # If any of these phrases appear in the final prompt, the default
        # behavioral pattern observation is contaminated.
        self.assertNotIn("step by step", out[0]["prompt"].lower())
        self.assertNotIn("think step", out[0]["prompt"].lower())

    def test_math500_no_cot_instruction(self):
        p = os.environ.get("PROJECT_ROOT", ".") + "/data/math500/test.parquet"
        if not os.path.exists(p):
            self.skipTest("math500 parquet missing")
        recs = load_parquet(p)[:3]
        out = C.math500_from_parquet(recs, self.d_cfg["benchmarks"]["math500"], self.prompts_cfg)
        for rec in out:
            self.assertNotIn("step by step", rec["prompt"].lower())
            self.assertNotIn("think step", rec["prompt"].lower())

    def test_ifeval(self):
        p = os.environ.get("PROJECT_ROOT", ".") + "/data/ifeval/test.parquet"
        if not os.path.exists(p):
            self.skipTest("ifeval parquet missing")
        recs = load_parquet(p)[:3]
        out = C.ifeval_from_parquet(recs, self.d_cfg["benchmarks"]["ifeval"], self.prompts_cfg)
        self.assertEqual(len(out), 3)
        md = out[0]["metadata"]
        self.assertIn("verifier_metadata", md)
        self.assertGreater(len(md["verifier_metadata"]["instruction_ids"]), 0)


if __name__ == "__main__":
    unittest.main()

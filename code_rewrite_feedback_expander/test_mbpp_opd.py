from __future__ import annotations

import json
import unittest

from code_rewrite_feedback_expander.mbpp_reward import extract_python_code, reward_func
from code_rewrite_feedback_expander.mbpp_to_opd_parquet import convert_row


class MBPPOPDDatasetTest(unittest.TestCase):
    def setUp(self) -> None:
        self.source = {
            "task_id": 1,
            "text": "Write a function add(a, b) that returns the sum.",
            "code": "def add(a, b):\n    return a + b",
            "test_list": ["assert add(2, 3) == 5"],
            "test_setup_code": "",
            "challenge_test_list": ["assert add(-1, 1) == 0"],
        }

    def test_conversion_produces_verl_schema(self) -> None:
        row = convert_row(self.source, split="train", row_index=0, include_challenge_tests=True)
        self.assertEqual(row["data_source"], "mbpp")
        self.assertEqual(row["prompt"][0]["role"], "user")
        payload = json.loads(row["reward_model"]["ground_truth"])
        self.assertEqual(len(payload["tests"]), 2)

    def test_reward_accepts_canonical_and_rejects_bad_code(self) -> None:
        row = convert_row(self.source, split="train", row_index=0, include_challenge_tests=True)
        ground_truth = row["reward_model"]["ground_truth"]
        good = reward_func(
            "mbpp",
            "<think>Use addition.</think>\n```python\ndef add(a, b):\n    return a + b\n```",
            ground_truth,
            row["extra_info"],
        )
        bad = reward_func(
            "mbpp",
            "```python\ndef add(a, b):\n    return a - b\n```",
            ground_truth,
            row["extra_info"],
        )
        self.assertTrue(good["passed"])
        self.assertFalse(bad["passed"])

    def test_code_extraction_rejects_thinking_text(self) -> None:
        code = extract_python_code("<think>analysis</think>\n```python\ndef f():\n    return 1\n```")
        self.assertEqual(code, "def f():\n    return 1")


if __name__ == "__main__":
    unittest.main()

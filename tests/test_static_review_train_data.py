import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from data_collection import prepare_static_review_train_data


class StaticReviewTrainDataTest(unittest.TestCase):
    def test_static_codecritic_samples_emit_correctness_score_contract(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            input_path = Path(tmp_dir) / "seed.jsonl"
            output_path = Path(tmp_dir) / "static.jsonl"
            sample = {
                "prepared_review_sample": True,
                "dataset_index": 7,
                "scoring_target": "codecritic_correctness",
                "score_scale": "codecritic_correctness_1_10",
                "target_correctness_score": 8.0,
                "problem": "Return x + 1.",
                "candidate_code": "def f(x):\n    return x + 1\n",
                "code_language": "python",
                "tests": ["assert f(1) == 2"],
                "tests_for_prompt": "assert f(1) == 2",
                "difficulty": "Easy",
                "source": "mbpp",
                "subset": "mbpp",
                "reference_scores": {"Correctness Verification": 8.0},
                "dimension_rubrics": {"Correctness Verification": "Correctness only."},
                "overall_score": 8,
                "correctness_label": "Correct",
                "objective": {
                    "public_test_pass_rate": 1.0,
                    "private_test_pass_rate": 1.0,
                    "full_test_pass_rate": 1.0,
                },
            }
            input_path.write_text(json.dumps(sample) + "\n", encoding="utf-8")

            argv = [
                "prepare_static_review_train_data.py",
                "--input",
                str(input_path),
                "--output",
                str(output_path),
            ]
            with mock.patch.object(sys, "argv", argv):
                prepare_static_review_train_data.main()

            item = json.loads(output_path.read_text(encoding="utf-8").splitlines()[0])
            response = item["response"][0]
            self.assertIn('"correctness_score": 8', response)
            self.assertNotIn('"axiom_grade"', response)
            self.assertEqual(item["score_scale"], "codecritic_correctness_1_10")
            self.assertEqual(item["parsed_score"], 8.0)
            self.assertEqual(item["target_correctness_score"], 8.0)
            self.assertEqual(item["q_value"], [1.0])
            self.assertIn("CodeCriticBench correctness scoring model", item["messages"][0]["content"])


if __name__ == "__main__":
    unittest.main()

import unittest
import json
import tempfile
from pathlib import Path

from mcts_math.review_utils import compute_pass_rate, load_codecriticbench_dataset, prepare_codecriticbench_sample


class ReviewSeedPreparationTest(unittest.TestCase):
    def test_compute_pass_rate_can_cap_assertions_for_seed_preparation(self) -> None:
        code = "def f(x):\n    return x"
        assertions = [
            "assert f(1) == 1",
            "assert f(2) == 2",
            "assert f(3) == 0",
        ]

        pass_rate = compute_pass_rate(code, assertions, max_assertions=2, timeout_seconds=1)

        self.assertEqual(pass_rate, 1.0)

    def test_stdin_style_tests_do_not_demote_codecritic_correct_labels(self) -> None:
        raw = {
            "question": "Read n and print n.",
            "answer": "n = int(input())\nprint(n)",
            "public_test": {"input": ["1\n7"]},
            "private_test": {"input": ["1\n9"]},
            "checklist_dimensions": ["Correctness Verification"],
            "checklist_scores": [8],
            "checklists": ["Does it solve the stated input/output task?"],
            "score": 8,
            "correctness": "Correct",
            "source": "debug",
            "subset": "debug",
        }

        sample = prepare_codecriticbench_sample(raw, max_objective_assertions_per_split=2, assertion_timeout_seconds=1)

        self.assertEqual(sample["objective"]["test_execution_kind"], "non_assertion")
        self.assertEqual(sample["objective"]["full_test_pass_rate"], 0.0)
        self.assertGreaterEqual(sample["axiom_target_grade"], 3)

    def test_loader_skips_active_excluded_prebuilt_axiom_sample(self) -> None:
        excluded = {
            "prepared_review_sample": True,
            "dataset_family": "axiom",
            "source": "axiom",
            "subset": "apps",
            "source_dataset": "apps.jsonl",
            "original_dataset_index": 420,
            "dataset_index": "apps:420",
            "problem": "Bad label.",
            "candidate_code": "print('bad')",
            "reference_scores": {"Correctness Verification": 10.0},
            "axiom_target_grade": 5,
        }
        kept = {
            **excluded,
            "original_dataset_index": 421,
            "dataset_index": "apps:421",
            "problem": "Keep this.",
        }
        exclusion = {
            "status": "active",
            "dataset_family": "axiom",
            "source": "axiom",
            "subset": "apps",
            "source_dataset": "apps.jsonl",
            "original_dataset_index": 420,
            "dataset_index": "apps:420",
        }
        with tempfile.TemporaryDirectory() as tmp:
            dataset_path = Path(tmp) / "seed.jsonl"
            exclusion_path = Path(tmp) / "excluded.jsonl"
            dataset_path.write_text(
                json.dumps(excluded) + "\n" + json.dumps(kept) + "\n",
                encoding="utf-8",
            )
            exclusion_path.write_text(json.dumps(exclusion) + "\n", encoding="utf-8")

            loaded = load_codecriticbench_dataset(str(dataset_path), start=0, limit=1, exclusion_path=exclusion_path)

        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0]["original_dataset_index"], 421)
        self.assertEqual(loaded[0]["problem"], "Keep this.")


if __name__ == "__main__":
    unittest.main()

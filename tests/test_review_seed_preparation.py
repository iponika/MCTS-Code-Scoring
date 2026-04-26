import unittest

from mcts_math.review_utils import compute_pass_rate, prepare_codecriticbench_sample


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


if __name__ == "__main__":
    unittest.main()

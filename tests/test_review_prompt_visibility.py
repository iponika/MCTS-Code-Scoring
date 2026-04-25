import unittest

from magicoder.preprocess_review_mcts_data import build_instruction


class ReviewPromptVisibilityTest(unittest.TestCase):
    def test_build_instruction_hides_dataset_tests_by_default(self) -> None:
        sample = {
            "problem": "Return x + 1.",
            "candidate_code": "def f(x):\n    return x + 1",
            "tests": ["assert f(1) == 2"],
            "language": "python",
        }

        instruction = build_instruction(sample, "Correctness Verification")

        self.assertNotIn("assert f(1) == 2", instruction)
        self.assertIn("No tests are available to the reviewer.", instruction)

    def test_build_instruction_can_show_tests_for_oracle_diagnostics(self) -> None:
        sample = {
            "problem": "Return x + 1.",
            "candidate_code": "def f(x):\n    return x + 1",
            "tests": ["assert f(1) == 2"],
            "language": "python",
        }

        instruction = build_instruction(sample, "Correctness Verification", show_tests_in_prompt=True)

        self.assertIn("assert f(1) == 2", instruction)


if __name__ == "__main__":
    unittest.main()

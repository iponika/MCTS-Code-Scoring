import unittest

from magicoder.batch_review_evaluator import build_parser


class BatchReviewEvaluatorCliTest(unittest.TestCase):
    def test_step_context_mode_defaults_to_instruction_context(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "--policy_model_path",
                "policy",
                "--input_record",
                "input.jsonl",
                "--record_indices",
                "0",
                "--output_dir",
                "out",
            ]
        )

        self.assertEqual(args.step_context_mode, "instruction_context")


if __name__ == "__main__":
    unittest.main()

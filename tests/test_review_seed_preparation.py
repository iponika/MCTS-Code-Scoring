import unittest

from mcts_math.review_utils import compute_pass_rate


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


if __name__ == "__main__":
    unittest.main()

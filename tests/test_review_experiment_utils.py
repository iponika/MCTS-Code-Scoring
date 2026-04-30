import unittest

from data_collection.review_experiment_utils import (
    direct_review_alignment_targets,
    normalize_step_counts,
    stepwise_variants,
)


class ReviewExperimentUtilsTest(unittest.TestCase):
    def test_normalize_step_counts_deduplicates_and_filters_non_positive(self) -> None:
        self.assertEqual(normalize_step_counts([2, 1, 2, 0, -3, 3]), [1, 2, 3])

    def test_stepwise_variants_build_expected_tags(self) -> None:
        self.assertEqual(
            stepwise_variants([2, 1]),
            [
                {"tag": "direct_stepwise_1step", "reasoning_steps": 1},
                {"tag": "direct_stepwise_2step", "reasoning_steps": 2},
            ],
        )

    def test_direct_review_alignment_targets_cap_policy_to_static_review_count(self) -> None:
        static_counts = {"policy": 96, "value": 0, "total": 96}
        direct_counts = {"policy": 288, "value": 144, "total": 432}

        self.assertEqual(
            direct_review_alignment_targets(static_counts, direct_counts),
            {
                "target_policy_count": 96,
                "target_value_count": -1,
                "target_total_count": -1,
                "static_policy_count": 96,
                "direct_policy_count": 288,
            },
        )

    def test_direct_review_alignment_targets_avoid_oversampling_when_direct_is_smaller(self) -> None:
        static_counts = {"policy": 120, "value": 0, "total": 120}
        direct_counts = {"policy": 80, "value": 32, "total": 112}

        self.assertEqual(
            direct_review_alignment_targets(static_counts, direct_counts),
            {
                "target_policy_count": 80,
                "target_value_count": -1,
                "target_total_count": -1,
                "static_policy_count": 120,
                "direct_policy_count": 80,
            },
        )


if __name__ == "__main__":
    unittest.main()

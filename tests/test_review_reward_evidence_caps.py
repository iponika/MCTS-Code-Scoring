import unittest

from mcts_math.review_utils import compute_review_reward


def sample_with_target_grade(grade: int) -> dict:
    return {
        "axiom_target_grade": grade,
        "objective": {"full_test_pass_rate": 1.0},
        "tests": ["assert f(1) == 2"],
        "correctness_label": "Correct",
        "candidate_code": "def f(x):\n    return x + 1",
    }


class ReviewRewardEvidenceCapsTest(unittest.TestCase):
    def test_correct_grade_with_wrong_test_failure_evidence_stays_above_boundary_error(self) -> None:
        review = (
            '<review>{"axiom_grade":5,"score":100,"verdict":"accept",'
            '"functional_correctness":true,"repair_effort":"none",'
            '"evidence_type":"provided_test_failure",'
            '"summary":"Correct code.",'
            '"evidence":["assert f(1) == 2"]}</review>'
        )

        reward, details = compute_review_reward("Correctness Verification", review, sample_with_target_grade(5))

        self.assertGreater(reward, -0.3)
        self.assertEqual(details["predicted_axiom_grade"], 5)
        self.assertLess(details["evidence_alignment"], 1.0)

    def test_wrong_low_grade_with_wrong_test_failure_evidence_remains_strongly_penalized(self) -> None:
        review = (
            '<review>{"axiom_grade":2,"score":40,"verdict":"major_issue",'
            '"functional_correctness":false,"repair_effort":"minor_functional",'
            '"evidence_type":"provided_test_failure",'
            '"summary":"Incorrectly claims a listed test fails.",'
            '"evidence":["assert f(1) == 2 fails"]}</review>'
        )

        reward, details = compute_review_reward("Correctness Verification", review, sample_with_target_grade(5))

        self.assertLessEqual(reward, -0.7)
        self.assertEqual(details["predicted_axiom_grade"], 2)


if __name__ == "__main__":
    unittest.main()

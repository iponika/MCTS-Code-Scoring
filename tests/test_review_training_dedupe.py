import json
import unittest

from magicoder.preprocess_review_mcts_data import convert_records


def terminal_node(tag: str, final_answer: str) -> tuple[str, dict]:
    details = {
        "parsed": json.loads(final_answer),
        "predicted_axiom_grade": json.loads(final_answer)["axiom_grade"],
        "target_axiom_grade": 5,
        "target_score": 100,
    }
    return tag, {
        "text": f"<review>\n{final_answer}\n</review>",
        "final_answer": final_answer,
        "target_dimension": "Correctness Verification",
        "reward_details": json.dumps(details),
        "q_value": 1.0,
    }


class ReviewTrainingDedupeTest(unittest.TestCase):
    def test_same_parent_semantic_duplicate_reviews_export_once(self) -> None:
        first_review = (
            '{"axiom_grade":5,"score":100,"verdict":"accept",'
            '"functional_correctness":true,"repair_effort":"none",'
            '"evidence_type":"uncertain","summary":"No defect is visible.",'
            '"evidence":["The implementation directly returns x + 1."]}'
        )
        duplicate_review = (
            '{"axiom_grade":5,"score":95,"verdict":"accept",'
            '"functional_correctness":true,"repair_effort":"none",'
            '"evidence_type":"uncertain","summary":"Equivalent review wording.",'
            '"evidence":["The return expression matches the requirement."]}'
        )
        first_tag, first_terminal = terminal_node("0.0.0.0", first_review)
        second_tag, second_terminal = terminal_node("0.0.0.1", duplicate_review)
        record = {
            "dataset_index": 7,
            "source": "unit",
            "subset": "unit",
            "problem": "Return x + 1.",
            "candidate_code": "def f(x):\n    return x + 1",
            "tests": [],
            "language": "python",
            "best_reviews_by_dimension": {"Correctness Verification": {"tag": first_tag}},
            "react": {
                "0": {},
                "0.0": {"target_dimension": "Correctness Verification"},
                "0.0.0": {
                    "text": "<step>\nCheck the return expression.\n</step>",
                    "target_dimension": "Correctness Verification",
                    "q_value": 1.0,
                },
                first_tag: first_terminal,
                second_tag: second_terminal,
            },
        }

        items, stats = convert_records(
            [record],
            policy_min_q=0.5,
            max_value_paths_per_dimension=0,
        )

        self.assertEqual(len(items), 1)
        self.assertEqual(stats["semantic_duplicate_reviews"], 1)


if __name__ == "__main__":
    unittest.main()

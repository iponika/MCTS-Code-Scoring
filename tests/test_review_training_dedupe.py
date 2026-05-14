import json
import unittest

from magicoder.preprocess_review_mcts_data import (
    attach_qwen_messages,
    convert_records,
    extract_response_segments,
)


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


def review_for_grade(grade: int) -> str:
    return json.dumps(
        {
            "axiom_grade": grade,
            "score": grade * 20,
            "verdict": "accept" if grade >= 3 else "reject",
            "functional_correctness": grade >= 3,
            "repair_effort": "none" if grade == 5 else "major_functional",
            "evidence_type": f"unit_grade_{grade}",
            "summary": f"Grade {grade} review.",
            "evidence": [f"Evidence for grade {grade}."],
        }
    )


def codecritic_terminal_node(tag: str, predicted_score: int, target_score: int = 5) -> tuple[str, dict]:
    final_answer = json.dumps(
        {
            "correctness_score": predicted_score,
            "dimension": "Correctness Verification",
            "evidence_type": "deduced_counterexample",
            "summary": f"Predicted score {predicted_score}.",
            "evidence": [f"Evidence for score {predicted_score}."],
        }
    )
    details = {
        "score_scale": "codecritic_correctness_1_10",
        "parsed": json.loads(final_answer),
        "predicted_correctness_score": predicted_score,
        "target_correctness_score": target_score,
    }
    return tag, {
        "text": f"<review>\n{final_answer}\n</review>",
        "final_answer": final_answer,
        "target_dimension": "Correctness Verification",
        "reward_details": json.dumps(details),
        "q_value": 1.0,
    }


class ReviewTrainingDedupeTest(unittest.TestCase):
    def test_extract_response_segments_preserves_prefix_reasoning_before_review(self) -> None:
        mixed = (
            "Trace x=1: the code returns 2.\n"
            "</think>\n\n"
            "<review>\n"
            '{"axiom_grade":5,"score":100,"verdict":"accept","functional_correctness":true,'
            '"repair_effort":"none","evidence_type":"deduced_counterexample",'
            '"summary":"The code returns x plus one.","evidence":["For x=1, code returns 2."]}'
            "\n</review>"
        )

        segments = extract_response_segments(mixed)

        self.assertEqual(len(segments), 2)
        self.assertFalse(segments[0].startswith("<step>"))
        self.assertIn("Trace x=1", segments[0])
        self.assertTrue(segments[1].startswith("<review>"))
        self.assertNotIn("</think>", "".join(segments))

    def test_convert_records_splits_mixed_terminal_reasoning_and_review(self) -> None:
        review = (
            '{"axiom_grade":5,"score":100,"verdict":"accept",'
            '"functional_correctness":true,"repair_effort":"none",'
            '"evidence_type":"deduced_counterexample","summary":"Code returns x plus one.",'
            '"evidence":["For x=1, code returns 2, matching the requirement."]}'
        )
        tag, terminal = terminal_node(
            "0.0.0.0",
            review,
        )
        terminal["text"] = (
            "Trace x=1: the code returns 2, matching the requirement.\n"
            "<review>\n"
            f"{review}\n"
            "</review>"
        )
        record = {
            "dataset_index": 6,
            "source": "unit",
            "subset": "unit",
            "problem": "Return x + 1.",
            "candidate_code": "def f(x):\n    return x + 1",
            "tests": [],
            "language": "python",
            "best_reviews_by_dimension": {"Correctness Verification": {"tag": tag}},
            "react": {
                "0": {},
                "0.0": {"target_dimension": "Correctness Verification"},
                tag: terminal,
            },
        }

        items, _stats = convert_records(
            [record],
            policy_min_q=0.5,
            max_value_paths_per_dimension=0,
        )

        self.assertEqual(len(items), 1)
        self.assertEqual(len(items[0]["response"]), 2)
        self.assertFalse(items[0]["response"][0].startswith("<step>"))
        self.assertTrue(items[0]["response"][1].startswith("<review>"))

    def test_convert_records_exports_qwen_messages_and_assistant_parts(self) -> None:
        review = (
            '{"axiom_grade":5,"score":100,"verdict":"accept",'
            '"functional_correctness":true,"repair_effort":"none",'
            '"evidence_type":"deduced_counterexample","summary":"Code returns x plus one.",'
            '"evidence":["For x=1, code returns 2, matching the requirement."]}'
        )
        tag, terminal = terminal_node("0.0.0.0", review)
        record = {
            "dataset_index": 10,
            "source": "unit",
            "subset": "unit",
            "problem": "Return x + 1.",
            "candidate_code": "def f(x):\n    return x + 1",
            "tests": [],
            "language": "python",
            "best_reviews_by_dimension": {"Correctness Verification": {"tag": tag}},
            "react": {
                "0": {},
                "0.0": {"target_dimension": "Correctness Verification"},
                "0.0.0": {
                    "text": "Trace x=1: code returns 2.",
                    "target_dimension": "Correctness Verification",
                    "q_value": 0.7,
                },
                tag: terminal,
            },
        }

        items, _stats = convert_records(
            [record],
            policy_min_q=0.5,
            max_value_paths_per_dimension=0,
        )

        item = items[0]
        self.assertEqual([message["role"] for message in item["messages"]], ["user", "assistant"])
        assistant_content = item["messages"][1]["content"]
        self.assertIn("<think>", assistant_content)
        self.assertIn("Trace x=1: code returns 2.", assistant_content)
        self.assertNotIn("<step>", assistant_content)
        self.assertIn("</think>\n\n<review>", assistant_content)
        self.assertEqual([part["type"] for part in item["assistant_parts"]], ["reasoning", "review"])
        self.assertEqual(
            [part["q_value"] for part in item["assistant_parts"]],
            item["q_value"],
        )

    def test_attach_qwen_messages_omits_empty_think_for_review_only_items(self) -> None:
        item = {
            "instruction": "Return x + 1.",
            "response": [
                "<review>\n"
                '{"axiom_grade":5,"score":100,"verdict":"accept","functional_correctness":true,'
                '"repair_effort":"none","evidence":["Matches the requirement."]}'
                "\n</review>"
            ],
            "q_value": [1.0],
        }

        attach_qwen_messages(item)

        assistant_content = item["messages"][1]["content"]
        self.assertTrue(assistant_content.startswith("<review>"))
        self.assertNotIn("<think>", assistant_content)
        self.assertEqual([part["type"] for part in item["assistant_parts"]], ["review"])

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

    def test_high_q_best_path_grade_mismatch_is_value_only(self) -> None:
        review = (
            '{"axiom_grade":4,"score":80,"verdict":"accept",'
            '"functional_correctness":true,"repair_effort":"minor_quality",'
            '"evidence_type":"uncertain","summary":"Correct but minor quality issue.",'
            '"evidence":["No concrete functional defect is visible."]}'
        )
        tag, terminal = terminal_node("0.0.0.0", review)
        details = json.loads(terminal["reward_details"])
        details["target_axiom_grade"] = 5
        details["target_score"] = 100
        terminal["reward_details"] = json.dumps(details)
        terminal["q_value"] = 0.82
        record = {
            "dataset_index": 8,
            "source": "unit",
            "subset": "unit",
            "problem": "Return x + 1.",
            "candidate_code": "def f(x):\n    return x + 1",
            "tests": [],
            "language": "python",
            "best_reviews_by_dimension": {"Correctness Verification": {"tag": tag}},
            "react": {
                "0": {},
                "0.0": {"target_dimension": "Correctness Verification"},
                "0.0.0": {
                    "text": "<step>\nCheck the return expression.\n</step>",
                    "target_dimension": "Correctness Verification",
                    "q_value": 0.82,
                },
                tag: terminal,
            },
        }

        items, stats = convert_records(
            [record],
            policy_min_q=0.5,
            max_value_paths_per_dimension=0,
        )

        self.assertEqual(len(items), 1)
        self.assertFalse(items[0]["train_lm"])
        self.assertTrue(items[0]["is_best_path"])
        self.assertEqual(stats["policy_grade_mismatch_paths"], 1)
        self.assertEqual(stats["value_only_paths"], 1)

    def test_high_q_best_path_exact_grade_remains_policy(self) -> None:
        review = (
            '{"axiom_grade":5,"score":100,"verdict":"accept",'
            '"functional_correctness":true,"repair_effort":"none",'
            '"evidence_type":"uncertain","summary":"No defect is visible.",'
            '"evidence":["The implementation directly returns x + 1."]}'
        )
        tag, terminal = terminal_node("0.0.0.0", review)
        record = {
            "dataset_index": 9,
            "source": "unit",
            "subset": "unit",
            "problem": "Return x + 1.",
            "candidate_code": "def f(x):\n    return x + 1",
            "tests": [],
            "language": "python",
            "best_reviews_by_dimension": {"Correctness Verification": {"tag": tag}},
            "react": {
                "0": {},
                "0.0": {"target_dimension": "Correctness Verification"},
                "0.0.0": {
                    "text": "<step>\nCheck the return expression.\n</step>",
                    "target_dimension": "Correctness Verification",
                    "q_value": 1.0,
                },
                tag: terminal,
            },
        }

        items, stats = convert_records(
            [record],
            policy_min_q=0.5,
            max_value_paths_per_dimension=0,
        )

        self.assertEqual(len(items), 1)
        self.assertTrue(items[0]["train_lm"])
        self.assertEqual(stats["policy_paths"], 1)

    def test_high_q_exact_grade_with_empty_final_evidence_is_value_only(self) -> None:
        review = (
            '{"axiom_grade":5,"score":100,"verdict":"accept",'
            '"functional_correctness":true,"repair_effort":"none",'
            '"evidence_type":"none","summary":"No defects.",'
            '"evidence":[]}'
        )
        tag, terminal = terminal_node("0.0.0.0", review)
        record = {
            "dataset_index": 11,
            "source": "unit",
            "subset": "unit",
            "problem": "Return x + 1.",
            "candidate_code": "def f(x):\n    return x + 1",
            "tests": [],
            "language": "python",
            "best_reviews_by_dimension": {"Correctness Verification": {"tag": tag}},
            "react": {
                "0": {},
                "0.0": {"target_dimension": "Correctness Verification"},
                "0.0.0": {
                    "text": "<step>\nTrace x=1: code returns 2.\n</step>",
                    "target_dimension": "Correctness Verification",
                    "q_value": 1.0,
                },
                tag: terminal,
            },
        }

        items, stats = convert_records(
            [record],
            policy_min_q=0.5,
            max_value_paths_per_dimension=0,
        )

        self.assertEqual(len(items), 1)
        self.assertFalse(items[0]["train_lm"])
        self.assertEqual(items[0]["policy_block_reason"], "weak_or_malformed_reasoning")
        self.assertEqual(stats["policy_reasoning_quality_blocked"], 1)

    def test_high_q_exact_grade_with_premature_review_step_is_value_only(self) -> None:
        review = (
            '{"axiom_grade":5,"score":100,"verdict":"accept",'
            '"functional_correctness":true,"repair_effort":"none",'
            '"evidence_type":"deduced_counterexample","summary":"Code returns x plus one.",'
            '"evidence":["For x=1, code returns 2, matching the requirement."]}'
        )
        tag, terminal = terminal_node("0.0.0.0", review)
        record = {
            "dataset_index": 12,
            "source": "unit",
            "subset": "unit",
            "problem": "Return x + 1.",
            "candidate_code": "def f(x):\n    return x + 1",
            "tests": [],
            "language": "python",
            "best_reviews_by_dimension": {"Correctness Verification": {"tag": tag}},
            "react": {
                "0": {},
                "0.0": {"target_dimension": "Correctness Verification"},
                "0.0.0": {
                    "text": "<step>\nPremature final review draft retained as a reasoning note, not as the final scored review: {\"axiom_grade\": 5}\n</step>",
                    "target_dimension": "Correctness Verification",
                    "q_value": 1.0,
                },
                tag: terminal,
            },
        }

        items, stats = convert_records(
            [record],
            policy_min_q=0.5,
            max_value_paths_per_dimension=0,
        )

        self.assertEqual(len(items), 1)
        self.assertFalse(items[0]["train_lm"])
        self.assertEqual(items[0]["policy_block_reason"], "weak_or_malformed_reasoning")
        self.assertEqual(stats["policy_reasoning_quality_blocked"], 1)

    def test_high_q_exact_grade_with_repeated_step_is_value_only(self) -> None:
        review = (
            '{"axiom_grade":5,"score":100,"verdict":"accept",'
            '"functional_correctness":true,"repair_effort":"none",'
            '"evidence_type":"deduced_counterexample","summary":"Code returns x plus one.",'
            '"evidence":["For x=1, code returns 2, matching the requirement."]}'
        )
        tag, terminal = terminal_node("0.0.0.0", review)
        record = {
            "dataset_index": 13,
            "source": "unit",
            "subset": "unit",
            "problem": "Return x + 1.",
            "candidate_code": "def f(x):\n    return x + 1",
            "tests": [],
            "language": "python",
            "best_reviews_by_dimension": {"Correctness Verification": {"tag": tag}},
            "react": {
                "0": {},
                "0.0": {
                    "text": "<step>\nTrace input x=1: the code returns 2, matching the requirement.\n</step>",
                    "target_dimension": "Correctness Verification",
                    "q_value": 1.0,
                },
                "0.0.0": {
                    "text": "<step>\nTrace x=1: code returns 2, matching the requirement.\n</step>",
                    "target_dimension": "Correctness Verification",
                    "q_value": 1.0,
                },
                tag: terminal,
            },
        }

        items, stats = convert_records(
            [record],
            policy_min_q=0.5,
            max_value_paths_per_dimension=0,
        )

        self.assertEqual(len(items), 1)
        self.assertFalse(items[0]["train_lm"])
        self.assertEqual(items[0]["policy_block_reason"], "weak_or_malformed_reasoning")
        self.assertEqual(stats["policy_reasoning_quality_blocked"], 1)

    def test_stage_value_labels_lift_promising_intermediate_step(self) -> None:
        review = (
            '{"axiom_grade":5,"score":100,"verdict":"accept",'
            '"functional_correctness":true,"repair_effort":"none",'
            '"evidence_type":"uncertain","summary":"No defect is visible.",'
            '"evidence":["The implementation directly returns x + 1."]}'
        )
        tag, terminal = terminal_node("0.0.0", review)
        record = {
            "dataset_index": 10,
            "source": "unit",
            "subset": "unit",
            "problem": "Return x + 1.",
            "candidate_code": "def f(x):\n    return x + 1",
            "tests": [],
            "language": "python",
            "best_reviews_by_dimension": {"Correctness Verification": {"tag": tag}},
            "react": {
                "0": {"target_dimension": "Correctness Verification"},
                "0.0": {
                    "text": "<step>\nFrame the full requirement before scoring.\n</step>",
                    "target_dimension": "Correctness Verification",
                    "q_value": -0.2,
                },
                tag: terminal,
            },
        }

        items, stats = convert_records(
            [record],
            policy_min_q=0.5,
            max_value_paths_per_dimension=0,
        )

        self.assertEqual(len(items), 1)
        item = items[0]
        self.assertEqual(item["raw_q_value"], [-0.2, 1.0])
        self.assertEqual(item["q_descendant_best"], [1.0, 1.0])
        self.assertGreater(item["q_value"][0], item["raw_q_value"][0])
        self.assertLess(item["q_value"][0], item["q_descendant_best"][0])
        self.assertEqual(item["q_value"][-1], 1.0)
        self.assertEqual(item["q_stage"], ["early", "review"])
        self.assertEqual(stats["stage_value_label_adjusted_items"], 1)

    def test_value_sampling_keeps_three_correct_and_three_incorrect_per_policy_record(self) -> None:
        policy_tag, policy = terminal_node("0.0", review_for_grade(5))
        react = {"0": {}, policy_tag: policy}
        for index in range(5):
            tag, node = terminal_node(f"{index + 1}.0", review_for_grade(5))
            node["q_value"] = 0.1 + index * 0.01
            react[tag] = node
        for index in range(5):
            tag, node = terminal_node(f"{index + 10}.0", review_for_grade(1))
            node["q_value"] = -0.1 - index * 0.01
            react[tag] = node
        record = {
            "dataset_index": 11,
            "source": "unit",
            "subset": "unit",
            "problem": "Return x + 1.",
            "candidate_code": "def f(x):\n    return x + 1",
            "tests": [],
            "language": "python",
            "best_reviews_by_dimension": {"Correctness Verification": {"tag": policy_tag}},
            "react": react,
        }

        items, stats = convert_records(
            [record],
            policy_min_q=0.5,
            max_value_paths_per_dimension=0,
            max_value_paths_per_sample_label=3,
            value_only_from_policy_records_only=True,
            stage_value_labels=False,
        )

        self.assertEqual(stats["policy_paths"], 1)
        self.assertEqual(stats["value_sampled_correct_paths"], 3)
        self.assertEqual(stats["value_sampled_incorrect_paths"], 3)
        self.assertEqual(stats["value_only_paths"], 6)
        self.assertEqual(len(items), 7)

    def test_codecritic_value_sampling_treats_within_policy_delta_as_high_quality_and_aligns_low_count(self) -> None:
        policy_tag, policy = codecritic_terminal_node("0.0", predicted_score=5, target_score=5)
        react = {"0": {}, policy_tag: policy}
        for index, score in enumerate([3, 6]):
            tag, node = codecritic_terminal_node(f"{index + 1}.0", predicted_score=score, target_score=5)
            node["q_value"] = 0.2 + index * 0.01
            react[tag] = node
        for index, score in enumerate([1, 2, 8, 9, 10]):
            tag, node = codecritic_terminal_node(f"{index + 10}.0", predicted_score=score, target_score=5)
            node["q_value"] = -0.2 - index * 0.01
            react[tag] = node
        record = {
            "dataset_index": 13,
            "source": "unit",
            "subset": "unit",
            "problem": "Return x + 1.",
            "candidate_code": "def f(x):\n    return x + 1",
            "tests": [],
            "language": "python",
            "react": react,
        }

        items, stats = convert_records(
            [record],
            policy_min_q=0.5,
            policy_max_grade_delta=2,
            max_value_paths_per_dimension=0,
            max_value_paths_per_sample_label=3,
            value_only_from_policy_records_only=True,
            stage_value_labels=False,
        )

        value_items = [item for item in items if not item.get("train_lm")]
        self.assertEqual(stats["policy_paths"], 1)
        self.assertEqual(stats["value_sampled_correct_paths"], 2)
        self.assertEqual(stats["value_sampled_incorrect_paths"], 2)
        self.assertEqual(stats["value_sample_dropped_incorrect_paths"], 3)
        self.assertEqual(stats["value_only_paths"], 4)
        self.assertEqual(len(value_items), 4)
        deltas = [int(item["parsed_score"] - item["target_correctness_score"]) for item in value_items]
        self.assertEqual(sum(1 for delta in deltas if abs(delta) <= 2), 2)
        self.assertEqual(sum(1 for delta in deltas if abs(delta) > 2), 2)

    def test_value_sampling_drops_value_only_record_without_policy(self) -> None:
        tag, node = terminal_node("0.0", review_for_grade(1))
        node["q_value"] = 0.1
        record = {
            "dataset_index": 12,
            "source": "unit",
            "subset": "unit",
            "problem": "Return x + 1.",
            "candidate_code": "def f(x):\n    return 0",
            "tests": [],
            "language": "python",
            "best_reviews_by_dimension": {"Correctness Verification": {"tag": tag}},
            "react": {"0": {}, tag: node},
        }

        items, stats = convert_records(
            [record],
            policy_min_q=0.5,
            max_value_paths_per_dimension=0,
            max_value_paths_per_sample_label=3,
            value_only_from_policy_records_only=True,
            stage_value_labels=False,
        )

        self.assertEqual(items, [])
        self.assertEqual(stats["records_without_policy"], 1)
        self.assertEqual(stats["value_only_paths_dropped_no_policy_record"], 1)


if __name__ == "__main__":
    unittest.main()

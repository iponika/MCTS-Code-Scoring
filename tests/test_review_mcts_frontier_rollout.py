import unittest

from omegaconf import OmegaConf

from mcts_math.agents import ReviewMCTS
from mcts_math.config import BaseConfig


def make_solver() -> ReviewMCTS:
    config = OmegaConf.structured(BaseConfig)
    config.mode = "mcts"
    config.model_dir = "."
    config.stop = ["</step>", "</review>"]
    config.max_depth = 3
    config.review_explore_depth = 1
    config.review_finalize_frontier_limit = 0
    config.disable_process_pool = True
    sample = {
        "reference_scores": {"Correctness Verification": 5.0},
        "dimension_rubrics": {"Correctness Verification": "Correctness only."},
        "candidate_code": "def f(x):\n    return x + 1",
        "code_language": "python",
        "tests_for_prompt": "No tests are available to the reviewer.",
        "axiom_target_grade": 5,
        "objective": {"full_test_pass_rate": 1.0},
        "tests": [],
        "correctness_label": "Correct",
    }
    return ReviewMCTS(config=config, question="Return x + 1.", review_sample=sample)


class ReviewMCTSFrontierRolloutTest(unittest.TestCase):
    def test_single_dimension_uses_root_as_review_start_node(self) -> None:
        solver = make_solver()
        self.assertEqual(solver.root.children, [])
        self.assertEqual(solver.root.state["target_dimension"], "Correctness Verification")
        self.assertEqual(solver.current_nodes, [solver.root])

    def test_exploration_stops_after_explore_depth(self) -> None:
        solver = make_solver()
        dimension_node = solver.root
        solver.create_child(
            "<step>\nCheck the return expression.\n</step>",
            {"action": "Check the return expression.", "action_input": "", "final_answer": ""},
            dimension_node,
            prior_prob=1.0,
            idx=0,
        )
        step_node = dimension_node.children[0]
        solver.current_nodes = [step_node]
        self.assertTrue(solver.should_generate_next())

        solver.create_child(
            "<step>\nCheck edge cases.\n</step>",
            {"action": "Check edge cases.", "action_input": "", "final_answer": ""},
            step_node,
            prior_prob=1.0,
            idx=0,
        )
        solver.current_nodes = [step_node.children[0]]
        self.assertFalse(solver.should_generate_next())

    def test_frontier_rollout_does_not_force_final_before_max_depth(self) -> None:
        solver = make_solver()
        dimension_node = solver.root
        solver.create_child(
            "<step>\nCheck the return expression.\n</step>",
            {"action": "Check the return expression.", "action_input": "", "final_answer": ""},
            dimension_node,
            prior_prob=1.0,
            idx=0,
        )

        self.assertTrue(solver.prepare_final_review_nodes())
        frontier = solver.current_nodes[0]
        self.assertFalse(frontier.state.get("force_final_review"))

    def test_frontier_rollout_forces_final_at_max_depth(self) -> None:
        solver = make_solver()
        dimension_node = solver.root
        first = dimension_node
        for depth in range(3):
            solver.create_child(
                f"<step>\nReasoning step {depth}.\n</step>",
                {"action": f"Reasoning step {depth}.", "action_input": "", "final_answer": ""},
                first,
                prior_prob=1.0,
                idx=0,
            )
            first = first.children[0]

        self.assertTrue(solver.prepare_final_review_nodes())
        frontier = solver.current_nodes[0]
        self.assertTrue(frontier.state.get("force_final_review"))

    def test_linear_rollout_accepts_natural_review_before_max_depth(self) -> None:
        solver = make_solver()
        dimension_node = solver.root
        solver.create_child(
            "<step>\nCheck the return expression.\n</step>",
            {"action": "Check the return expression.", "action_input": "", "final_answer": ""},
            dimension_node,
            prior_prob=1.0,
            idx=0,
        )
        self.assertTrue(solver.prepare_final_review_nodes())
        frontier = solver.current_nodes[0]

        review = (
            '{"axiom_grade":5,"score":100,"verdict":"accept",'
            '"functional_correctness":true,"repair_effort":"none",'
            '"evidence_type":"uncertain","summary":"No defect is visible.",'
            '"evidence":["The implementation directly returns x + 1."]}'
        )
        solver.create_child(
            f"<review>\n{review}\n</review>",
            {"action": "", "action_input": "", "final_answer": review},
            frontier,
            prior_prob=1.0,
            idx=0,
        )

        review_node = frontier.children[0]
        self.assertTrue(review_node.is_terminal)
        self.assertEqual(review_node.state["final_answer"], review)
        self.assertGreater(review_node.q_value(), 0)

    def test_duplicate_semantic_review_siblings_are_skipped(self) -> None:
        solver = make_solver()
        dimension_node = solver.root
        solver.create_child(
            "<step>\nCheck the return expression.\n</step>",
            {"action": "Check the return expression.", "action_input": "", "final_answer": ""},
            dimension_node,
            prior_prob=1.0,
            idx=0,
        )
        self.assertTrue(solver.prepare_final_review_nodes())
        frontier = solver.current_nodes[0]

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
        solver.create_child(
            f"<review>\n{first_review}\n</review>",
            {"action": "", "action_input": "", "final_answer": first_review},
            frontier,
            prior_prob=1.0,
            idx=0,
        )
        solver.create_child(
            f"<review>\n{duplicate_review}\n</review>",
            {"action": "", "action_input": "", "final_answer": duplicate_review},
            frontier,
            prior_prob=1.0,
            idx=1,
        )

        self.assertEqual(len(frontier.children), 1)


if __name__ == "__main__":
    unittest.main()

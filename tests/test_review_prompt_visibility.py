import unittest

from omegaconf import OmegaConf

from mcts_math.agents.utils import review_prompt_wrap, review_step_result_unwrap
from mcts_math.config import BaseConfig
from magicoder.preprocess_review_mcts_data import build_instruction
from magicoder.prompt_template import review_prompt_for_response
from magicoder.review_evaluator import prompt_for_dimension


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

    def test_step_prompt_hides_final_review_json_template(self) -> None:
        config = OmegaConf.structured(BaseConfig)
        prompt = review_prompt_wrap(
            "Return x + 1.",
            "",
            config,
            {
                "target_dimension": "Correctness Verification",
                "dimension_rubric": "Correctness only.",
                "candidate_code": "def f(x):\n    return x + 1",
                "code_language": "python",
                "tests_for_prompt": "No tests are available to the reviewer.",
                "force_final_review": False,
            },
        )

        self.assertIn("<step>", prompt)
        self.assertIn("Do not output <review> yet", prompt)
        self.assertNotIn("Structured final review format", prompt)
        self.assertNotIn('"axiom_grade"', prompt)
        self.assertNotIn("exceptionally intelligent", prompt)
        self.assertNotIn("coding assistant", prompt)
        self.assertNotIn("problem-solving plan", prompt)
        self.assertNotIn("<code>", prompt)

    def test_final_prompt_shows_final_review_json_template(self) -> None:
        config = OmegaConf.structured(BaseConfig)
        prompt = review_prompt_wrap(
            "Return x + 1.",
            "Trace the return expression.",
            config,
            {
                "target_dimension": "Correctness Verification",
                "dimension_rubric": "Correctness only.",
                "candidate_code": "def f(x):\n    return x + 1",
                "code_language": "python",
                "tests_for_prompt": "No tests are available to the reviewer.",
                "force_final_review": True,
            },
        )

        self.assertIn("Structured final review format", prompt)
        self.assertIn("<review>", prompt)
        self.assertIn('"axiom_grade"', prompt)
        self.assertNotIn("exceptionally intelligent", prompt)
        self.assertNotIn("coding assistant", prompt)

    def test_qwen_thinking_mode_appends_soft_switch(self) -> None:
        config = OmegaConf.structured(BaseConfig)
        config.qwen_thinking_mode = "think"
        prompt = review_prompt_wrap(
            "Return x + 1.",
            "",
            config,
            {
                "target_dimension": "Correctness Verification",
                "dimension_rubric": "Correctness only.",
                "candidate_code": "def f(x):\n    return x + 1",
                "code_language": "python",
                "tests_for_prompt": "No tests are available to the reviewer.",
                "force_final_review": False,
            },
        )

        self.assertTrue(prompt.rstrip().endswith("/think"))

    def test_native_thinking_final_prompt_uses_no_think(self) -> None:
        config = OmegaConf.structured(BaseConfig)
        config.qwen_thinking_mode = "think"
        config.review_native_thinking_steps = True
        prompt = review_prompt_wrap(
            "Return x + 1.",
            "<step>\nnative_think: trace return\n</step>",
            config,
            {
                "target_dimension": "Correctness Verification",
                "dimension_rubric": "Correctness only.",
                "candidate_code": "def f(x):\n    return x + 1",
                "code_language": "python",
                "tests_for_prompt": "No tests are available to the reviewer.",
                "force_final_review": True,
            },
        )

        self.assertTrue(prompt.rstrip().endswith("/no_think"))

    def test_native_think_block_becomes_step_node_text(self) -> None:
        step_text, parsed = review_step_result_unwrap("<think>\nTrace f(1): returns 2.\n</think>")

        self.assertIn("native_think", step_text)
        self.assertIn("<step>", step_text)
        self.assertEqual(parsed["action"], step_text)
        self.assertEqual(parsed["final_answer"], "")

    def test_stepwise_eval_step_prompt_marks_prior_steps_as_completed(self) -> None:
        sample = {
            "problem": "Return x + 1.",
            "candidate_code": "def f(x):\n    return x + 1",
            "tests": ["assert f(1) == 2"],
            "language": "python",
        }
        prompt = prompt_for_dimension(
            sample,
            "Correctness Verification",
            partial_response="<step>\nstatic_logic_check: The function returns x + 1 directly.\n</step>\n",
            force_final=False,
        )

        self.assertIn("completed previous <step> blocks", prompt)
        self.assertIn("Continue from the last completed step", prompt)
        self.assertIn("Do not output <review> yet", prompt)
        self.assertNotIn("unless the review is already ready", prompt)
        self.assertNotIn('"axiom_grade"', prompt)
        self.assertNotIn("Final review format", prompt)
        self.assertNotIn("exceptionally intelligent", prompt)

    def test_stepwise_eval_final_prompt_shows_review_format(self) -> None:
        sample = {
            "problem": "Return x + 1.",
            "candidate_code": "def f(x):\n    return x + 1",
            "tests": ["assert f(1) == 2"],
            "language": "python",
        }
        prompt = prompt_for_dimension(
            sample,
            "Correctness Verification",
            partial_response="<step>\nstatic_logic_check: The function returns x + 1 directly.\n</step>\n",
            force_final=True,
        )

        self.assertIn("completed previous <step> blocks", prompt)
        self.assertIn("Start your next output with <review>", prompt)
        self.assertIn('"axiom_grade"', prompt)

    def test_review_training_prompt_matches_response_shape(self) -> None:
        instruction = "Scoring target: assess candidate code correctness.\n\nTask description:\nReturn x + 1."

        final_prompt = review_prompt_for_response(instruction, "<review>\n{\"axiom_grade\": 5}\n</review>")
        step_prompt = review_prompt_for_response(instruction, "<step>\ntrace_requirement: Check return value.\n</step>")

        self.assertIn("Output exactly one JSON object wrapped in <review> tags", final_prompt)
        self.assertNotIn("Reasoning format: write concise <step> blocks", final_prompt)
        self.assertIn("Reasoning format: write concise <step> blocks", step_prompt)
        self.assertIn("then finish with exactly one <review> JSON block", step_prompt)

    def test_review_prompt_template_module_does_not_export_unused_legacy_prompts(self) -> None:
        import magicoder.prompt_template as prompt_template

        for name in [
            "SPHE_PROMPT",
            "SPMP_PROMPT",
            "COTHE_PROMPT",
            "COTMP_PROMPT",
            "QWEN_DIRECT_PROMPT",
            "QWEN_STEP_PROMPT",
            "MAGICODER_PROMPT",
        ]:
            self.assertFalse(hasattr(prompt_template, name), name)


if __name__ == "__main__":
    unittest.main()

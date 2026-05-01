import unittest

from omegaconf import OmegaConf

from mcts_math.agents.utils import review_prompt_wrap, review_step_result_unwrap
from mcts_math.config import BaseConfig
from magicoder.preprocess_review_mcts_data import build_instruction
from magicoder.prompt_template import review_prompt_for_response
from magicoder.review_evaluator import prompt_for_dimension
from direct_bootstrap_review import build_prompt as build_direct_bootstrap_prompt
from direct_bootstrap_review import direct_bootstrap_stop_tokens
from mcts_math.llms.local_llms import chat_messages_for_prompt


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
        self.assertIn("Do not restate the whole task", prompt)
        self.assertIn("one concise evidence increment", prompt)
        self.assertNotIn("as long as needed", prompt)
        self.assertNotIn("Structured final review format", prompt)
        self.assertNotIn('"axiom_grade"', prompt)
        self.assertNotIn("under 40 words", prompt)
        self.assertNotIn("{mode_instruction}", prompt)
        self.assertNotIn("2. Use completed previous steps", prompt)
        self.assertIn("minor tweaking", prompt)
        self.assertIn("Examples are illustrative, not exhaustive criteria", prompt)
        self.assertIn("for example, adding a boundary check", prompt)
        self.assertIn("for example, rewriting an entire code block", prompt)
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
        self.assertIn("reconcile supported previous-step evidence", prompt)
        self.assertIn("cannot silently contradict it", prompt)
        self.assertNotIn("{mode_instruction}", prompt)
        self.assertNotIn("2. Use completed previous steps", prompt)
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
        step_text, parsed = review_step_result_unwrap("<think>\nOkay, let's see. Trace f(1): returns 2.\n</think>")

        self.assertIn("native_think", step_text)
        self.assertIn("Trace f(1): returns 2.", step_text)
        self.assertIn("<step>", step_text)
        self.assertEqual(parsed["action"], step_text)
        self.assertEqual(parsed["final_answer"], "")

    def test_native_think_block_keeps_evidence_despite_native_intro(self) -> None:
        step_text, _ = review_step_result_unwrap(
            "<think>\nOkay, let's continue analyzing the candidate code. The pair-count logic overcounts evidence.\n</think>"
        )

        self.assertIn("The pair-count logic overcounts evidence.", step_text)

    def test_native_think_block_drops_meta_sentences(self) -> None:
        step_text, _ = review_step_result_unwrap(
            "<think>\n"
            "Okay, let me try to figure this out. "
            "The user wants me to continue evidence gathering for the AXIOM score. "
            "The previous step was about the sorting split. "
            "The split-and-concatenate logic returns all integers before all strings, which is only correct if that grouping is the required order."
            "\n</think>"
        )

        self.assertNotIn("The user wants me", step_text)
        self.assertNotIn("previous step", step_text.lower())
        self.assertIn("split-and-concatenate logic", step_text)

    def test_native_think_block_keeps_complete_evidence_increment(self) -> None:
        step_text, _ = review_step_result_unwrap(
            "<think>\n"
            "The task is to determine whether the implementation is correct. "
            "The candidate code is supposed to solve the problem. "
            "The pair-count logic overcounts because matching pair differences do not guarantee one compatible arithmetic sequence. "
            "This gives an unsupported low replacement count for mixed pairs"
            "</think>"
        )

        self.assertNotIn("The task is to determine", step_text)
        self.assertNotIn("supposed to solve", step_text)
        self.assertIn("The pair-count logic overcounts", step_text)
        self.assertTrue(step_text.rstrip().endswith(".\n</step>"))

    def test_data_collection_prompt_continues_after_response_prefix(self) -> None:
        config = OmegaConf.structured(BaseConfig)
        partial = "<step>\nstatic_logic_check: The function returns x + 1 directly.\n</step>"
        prompt = review_prompt_wrap(
            "Return x + 1.",
            partial,
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

        response_tail = prompt.split("@@ Response", 1)[1]
        self.assertIn(partial, response_tail)
        self.assertIn("Continue from the last completed step", prompt)

    def test_direct_bootstrap_final_prompt_has_consistency_rule(self) -> None:
        config = OmegaConf.structured(BaseConfig)
        sample = {
            "question": "Return x + 1.",
            "candidate_code": "def f(x):\n    return x + 1",
            "code_language": "python",
            "tests_for_prompt": "No tests are available to the reviewer.",
        }

        prompt = build_direct_bootstrap_prompt(
            sample,
            "Correctness Verification",
            config,
            partial_solution="<step>\nstatic_logic_check: The function returns x + 1 directly.\n</step>",
            force_final_review=True,
        )

        self.assertIn("reconcile supported previous-step evidence", prompt)
        self.assertIn("very first non-whitespace characters", prompt)
        self.assertIn("continue immediately with a JSON object", prompt)
        self.assertIn("@@ Response\n<step>\nstatic_logic_check: The function returns x + 1 directly.\n</step>\n<review>\n{", prompt)

    def test_thinking_configs_stop_at_native_think_close(self) -> None:
        for path in [
            "data_collection/configs/mcts_code_review_qwen3_4b.yaml",
            "data_collection/configs/mcts_code_review_qwen3_4b_thinking.yaml",
            "data_collection/configs/mcts_code_review_deepseek_r1_distill_qwen_7b.yaml",
        ]:
            config = OmegaConf.load(path)
            self.assertIn("</think>", list(config.stop), path)

    def test_direct_stepwise_stop_tokens_include_native_think_close(self) -> None:
        self.assertEqual(direct_bootstrap_stop_tokens("stepwise"), ["</think>", "</step>"])
        self.assertEqual(direct_bootstrap_stop_tokens("review"), ["</review>"])

    def test_chat_template_treats_response_prefix_as_assistant_history(self) -> None:
        prompt = (
            "Do the review.\n\n"
            "@@ Response\n"
            "<step>\nstatic_logic_check: The function returns x + 1 directly.\n</step>\n\n"
            "/think"
        )

        messages = chat_messages_for_prompt(prompt)

        self.assertEqual([message["role"] for message in messages], ["user", "assistant"])
        self.assertIn("@@ Response", messages[0]["content"])
        self.assertIn("<step>", messages[1]["content"])
        self.assertNotIn("/think", messages[1]["content"])

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
        self.assertNotIn("under 40 words", prompt)
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
        self.assertIn("reconcile supported previous-step evidence", prompt)
        self.assertIn("cannot silently contradict it", prompt)
        self.assertIn('"axiom_grade"', prompt)

    def test_review_training_prompt_matches_response_shape(self) -> None:
        instruction = "Scoring target: assess candidate code correctness.\n\nTask description:\nReturn x + 1."

        final_prompt = review_prompt_for_response(instruction, "<review>\n{\"axiom_grade\": 5}\n</review>")
        step_prompt = review_prompt_for_response(instruction, "<step>\ntrace_requirement: Check return value.\n</step>")

        self.assertIn("Output exactly one JSON object wrapped in <review> tags", final_prompt)
        self.assertIn("Current generation mode: final AXIOM scoring", final_prompt)
        self.assertIn("reconcile supported previous-step evidence", final_prompt)
        self.assertIn("cannot silently contradict it", final_prompt)
        self.assertNotIn("stepwise evidence-and-review training", final_prompt)
        self.assertIn("Current generation mode: stepwise evidence-and-review training", step_prompt)
        self.assertIn("finish with exactly one <review> JSON block", step_prompt)
        self.assertIn("each <step> should add one concrete", step_prompt)
        self.assertIn("Do not restate the whole task", step_prompt)
        self.assertNotIn("as long as needed", step_prompt)
        self.assertIn("reconcile supported step evidence", step_prompt)
        self.assertIn("minor tweaking", final_prompt)
        self.assertIn("major refactoring", final_prompt)
        self.assertIn("Examples are illustrative, not exhaustive criteria", final_prompt)
        self.assertIn("for example, an off-by-one error", final_prompt)
        self.assertIn("for example, replacing the required algorithm", final_prompt)

    def test_axiom_grade_descriptions_keep_example_wording_soft(self) -> None:
        from magicoder.axiom_scoring import AXIOM_GRADE_DESCRIPTIONS

        self.assertIn("for example", AXIOM_GRADE_DESCRIPTIONS[4])
        self.assertIn("for example", AXIOM_GRADE_DESCRIPTIONS[3])
        self.assertIn("for example", AXIOM_GRADE_DESCRIPTIONS[2])
        self.assertIn("for example", AXIOM_GRADE_DESCRIPTIONS[1])
        self.assertIn("unrelated task", AXIOM_GRADE_DESCRIPTIONS[0])
        self.assertNotIn("must be", AXIOM_GRADE_DESCRIPTIONS[2])

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

    def test_data_collection_review_prompt_module_does_not_export_codegen_prompts(self) -> None:
        import mcts_math.prompts.prompt_sft as prompt_sft

        for name in [
            "DEEPSEEK_PROMPT",
            "DEEPSEEK_LCB_PROMPT",
            "QWEN_DIRECT_PROMPT",
            "QWEN_STEP_PROMPT",
        ]:
            self.assertFalse(hasattr(prompt_sft, name), name)


if __name__ == "__main__":
    unittest.main()

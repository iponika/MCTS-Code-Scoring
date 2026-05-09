import unittest
from types import SimpleNamespace

from omegaconf import OmegaConf

from mcts_math.agents.utils import review_prompt_wrap, review_step_result_unwrap
from mcts_math.config import BaseConfig
from magicoder.preprocess_review_mcts_data import build_instruction
from magicoder.prompt_template import review_prompt_for_response
from magicoder.review_evaluator import build_chat_eval_prompt, prompt_for_dimension
from direct_bootstrap_review import build_prompt as build_direct_bootstrap_prompt
from direct_bootstrap_review import direct_bootstrap_stop_tokens
from mcts_math.llms.local_llms import chat_messages_for_prompt, maybe_apply_chat_template


class ReviewPromptVisibilityTest(unittest.TestCase):
    class _FakeChatTokenizer:
        def __init__(self) -> None:
            self.last_messages = None
            self.last_kwargs = None

        def apply_chat_template(self, messages, **kwargs):
            self.last_messages = messages
            self.last_kwargs = kwargs
            chunks = []
            for message in messages:
                chunks.append(f"[{message['role']}]\n{message['content']}")
            rendered = "\n\n".join(chunks)
            if kwargs.get("add_generation_prompt"):
                rendered += "\n\n[assistant]\n"
            return rendered

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

        self.assertIn("This is an intermediate turn of a multi-step code review", prompt)
        self.assertIn('first output token must be "1. "', prompt)
        self.assertIn("Your analysis object is as follows:", prompt)
        self.assertNotIn("<step>", prompt)
        self.assertNotIn('"step_type"', prompt)
        self.assertNotIn("as long as needed", prompt)
        self.assertNotIn("Structured final review format", prompt)
        self.assertNotIn('"axiom_grade"', prompt)
        self.assertNotIn("under 40 words", prompt)
        self.assertNotIn("{mode_instruction}", prompt)
        self.assertNotIn("2. Use completed previous steps", prompt)
        self.assertIn("minor tweaking", prompt)
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
        self.assertIn("reconcile supported previous reasoning evidence", prompt)
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
            "trace return",
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

    def test_native_think_block_becomes_reasoning_node_text(self) -> None:
        reasoning_text, parsed = review_step_result_unwrap("<think>\nOkay, let's see. Trace f(1): returns 2.\n</think>")

        self.assertIn("Trace f(1): returns 2.", reasoning_text)
        self.assertNotIn("<step>", reasoning_text)
        self.assertEqual(parsed["action"], reasoning_text)
        self.assertEqual(parsed["final_answer"], "")

    def test_native_think_block_keeps_evidence_despite_native_intro(self) -> None:
        reasoning_text, _ = review_step_result_unwrap(
            "<think>\nOkay, let's continue analyzing the candidate code. The pair-count logic overcounts evidence.\n</think>"
        )

        self.assertIn("The pair-count logic overcounts evidence.", reasoning_text)

    def test_native_think_block_drops_meta_sentences(self) -> None:
        reasoning_text, _ = review_step_result_unwrap(
            "<think>\n"
            "Okay, let me try to figure this out. "
            "The user wants me to continue evidence gathering for the AXIOM score. "
            "The previous step was about the sorting split. "
            "The split-and-concatenate logic returns all integers before all strings, which is only correct if that grouping is the required order."
            "\n</think>"
        )

        self.assertNotIn("The user wants me", reasoning_text)
        self.assertNotIn("previous step", reasoning_text.lower())
        self.assertIn("split-and-concatenate logic", reasoning_text)

    def test_native_think_block_keeps_complete_evidence_increment(self) -> None:
        reasoning_text, _ = review_step_result_unwrap(
            "<think>\n"
            "The task is to determine whether the implementation is correct. "
            "The candidate code is supposed to solve the problem. "
            "The pair-count logic overcounts because matching pair differences do not guarantee one compatible arithmetic sequence. "
            "This gives an unsupported low replacement count for mixed pairs"
            "</think>"
        )

        self.assertNotIn("The task is to determine", reasoning_text)
        self.assertNotIn("supposed to solve", reasoning_text)
        self.assertIn("The pair-count logic overcounts", reasoning_text)
        self.assertTrue(reasoning_text.rstrip().endswith("."))

    def test_data_collection_prompt_continues_after_response_prefix(self) -> None:
        config = OmegaConf.structured(BaseConfig)
        partial = "static_logic_check: The function returns x + 1 directly."
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
        self.assertIn("assistant history", prompt)
        self.assertIn("Generate one concise native reasoning note", prompt)

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
            partial_solution="static_logic_check: The function returns x + 1 directly.",
            force_final_review=True,
        )

        self.assertIn("This is the final turn of a multi-step code review", prompt)
        self.assertIn("Output must be exactly one JSON object wrapped in <review> tags", prompt)
        self.assertIn("evidence_type must be one of", prompt)
        self.assertIn("repair_effort must match the selected AXIOM grade", prompt)
        self.assertIn("reconcile supported previous reasoning evidence", prompt)
        self.assertNotIn('"score"', prompt)
        self.assertNotIn('"verdict"', prompt)
        self.assertIn("static_logic_check: The function returns x + 1 directly.", prompt)
        self.assertIn("You don't have to analyze code by yourself.", prompt)
        self.assertIn("Do not continue the numbered analysis notes.", prompt)
        self.assertIn("Start your very first output token with <review>", prompt)
        self.assertEqual(prompt.split("@@ Response", 1)[1].strip(), "")

    def test_direct_bootstrap_stepwise_final_moves_prior_steps_into_instruction(self) -> None:
        config = OmegaConf.structured(BaseConfig)
        sample = {
            "question": "Return x + 1.",
            "candidate_code": "def f(x):\n    return x + 1",
            "code_language": "python",
            "tests_for_prompt": "No tests are available to the reviewer.",
        }
        prior_step = "static_logic_check: The function returns x + 1 directly."

        prompt = build_direct_bootstrap_prompt(
            sample,
            "Correctness Verification",
            config,
            partial_solution=prior_step,
            force_final_review=True,
            steps_as_instruction_context=True,
        )

        self.assertIn("You don't have to analyze code by yourself.", prompt)
        self.assertIn("Do not continue the numbered analysis notes.", prompt)
        self.assertIn("static_logic_check: The function returns x + 1 directly.", prompt)
        response_tail = prompt.split("@@ Response", 1)[1]
        self.assertNotIn(prior_step, response_tail)
        self.assertEqual(response_tail.strip(), "")

    def test_direct_bootstrap_stepwise_step_can_move_prior_steps_into_instruction(self) -> None:
        config = OmegaConf.structured(BaseConfig)
        sample = {
            "question": "Return x + 1.",
            "candidate_code": "def f(x):\n    return x + 1",
            "code_language": "python",
            "tests_for_prompt": "No tests are available to the reviewer.",
        }
        prior_step = "static_logic_check: The function returns x + 1 directly."

        prompt = build_direct_bootstrap_prompt(
            sample,
            "Correctness Verification",
            config,
            partial_solution=prior_step,
            force_final_review=False,
            steps_as_instruction_context=True,
        )

        self.assertIn("Previous analysis notes:", prompt)
        self.assertIn("static_logic_check: The function returns x + 1 directly.", prompt)
        response_tail = prompt.split("@@ Response", 1)[1]
        self.assertNotIn(prior_step, response_tail)
        self.assertIn('first output token must be "1. "', prompt)
        self.assertIn("Your analysis object is as follows:", prompt)

    def test_thinking_configs_stop_at_native_think_close(self) -> None:
        for path in [
            "data_collection/configs/mcts_code_review_qwen3_4b.yaml",
            "data_collection/configs/mcts_code_review_qwen3_4b_thinking.yaml",
            "data_collection/configs/mcts_code_review_deepseek_r1_distill_qwen_7b.yaml",
        ]:
            config = OmegaConf.load(path)
            self.assertIn("</think>", list(config.stop), path)
            self.assertNotIn("</step>", list(config.stop), path)

    def test_direct_stepwise_stop_tokens_include_native_think_close(self) -> None:
        self.assertEqual(direct_bootstrap_stop_tokens("stepwise"), ["</think>"])
        self.assertEqual(direct_bootstrap_stop_tokens("review"), ["</review>"])

    def test_chat_template_treats_response_prefix_as_assistant_history(self) -> None:
        prompt = (
            "Do the review.\n\n"
            "@@ Response\n"
            "static_logic_check: The function returns x + 1 directly.\n\n"
            "/think"
        )

        messages = chat_messages_for_prompt(prompt)

        self.assertEqual([message["role"] for message in messages], ["user", "assistant"])
        self.assertIn("@@ Response", messages[0]["content"])
        self.assertIn("static_logic_check", messages[1]["content"])
        self.assertNotIn("/think", messages[1]["content"])

    def test_mcts_chat_template_appends_response_prefix_to_active_assistant_turn(self) -> None:
        prompt = (
            "Do the review.\n\n"
            "@@ Response\n"
            "static_logic_check: The function returns x + 1 directly.\n\n"
            "/no_think"
        )
        tokenizer = self._FakeChatTokenizer()

        class _FakeEngine:
            def get_tokenizer(self):
                return tokenizer

        rendered = maybe_apply_chat_template(
            [prompt],
            _FakeEngine(),
            SimpleNamespace(use_chat_template=True, chat_template_enable_thinking=True),
        )[0]

        self.assertEqual(len(tokenizer.last_messages), 1)
        self.assertEqual(tokenizer.last_messages[0]["role"], "user")
        self.assertIn("@@ Response", tokenizer.last_messages[0]["content"])
        self.assertNotIn("static_logic_check", tokenizer.last_messages[0]["content"])
        self.assertEqual(tokenizer.last_kwargs["enable_thinking"], False)
        self.assertTrue(rendered.rstrip().endswith("static_logic_check: The function returns x + 1 directly."))
        self.assertEqual(rendered.count("[assistant]"), 1)

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
            partial_response="static_logic_check: The function returns x + 1 directly.\n",
            force_final=False,
        )

        self.assertIn("prior thinking history", prompt)
        self.assertIn("This is an intermediate turn of a multi-step code review", prompt)
        self.assertIn("@@ Response\nstatic_logic_check: The function returns x + 1 directly.\n", prompt)
        self.assertIn('first output token must be "1. "', prompt)
        self.assertNotIn("under 40 words", prompt)
        self.assertNotIn("unless the review is already ready", prompt)
        self.assertNotIn('"axiom_grade"', prompt)
        self.assertNotIn("Final review format", prompt)
        self.assertNotIn("exceptionally intelligent", prompt)

    def test_stepwise_eval_step_prompt_can_move_prior_steps_into_instruction(self) -> None:
        sample = {
            "problem": "Return x + 1.",
            "candidate_code": "def f(x):\n    return x + 1",
            "tests": ["assert f(1) == 2"],
            "language": "python",
        }
        prior_step = "static_logic_check: The function returns x + 1 directly.\n"
        prompt = prompt_for_dimension(
            sample,
            "Correctness Verification",
            partial_response=prior_step,
            force_final=False,
            step_context_mode="instruction_context",
        )

        self.assertIn("Previous analysis notes:", prompt)
        self.assertIn("static_logic_check: The function returns x + 1 directly.", prompt)
        response_tail = prompt.split("@@ Response", 1)[1]
        self.assertNotIn(prior_step.strip(), response_tail)
        self.assertNotIn("Continue from the last completed step", prompt)
        self.assertIn('first output token must be "1. "', prompt)

    def test_chat_eval_prompt_respects_instruction_context(self) -> None:
        sample = {
            "problem": "Return x + 1.",
            "candidate_code": "def f(x):\n    return x + 1",
            "tests": ["assert f(1) == 2"],
            "language": "python",
        }
        prior_step = "static_logic_check: The function returns x + 1 directly.\n"
        tokenizer = self._FakeChatTokenizer()

        rendered = build_chat_eval_prompt(
            tokenizer,
            sample,
            "Correctness Verification",
            partial_response=prior_step,
            force_final=False,
            step_context_mode="instruction_context",
            show_tests_in_prompt=False,
        )

        self.assertEqual(len(tokenizer.last_messages), 1)
        self.assertEqual(tokenizer.last_messages[0]["role"], "user")
        self.assertIn("Previous analysis notes:", tokenizer.last_messages[0]["content"])
        self.assertIn("static_logic_check: The function returns x + 1 directly.", tokenizer.last_messages[0]["content"])
        self.assertIn("@@ Response", tokenizer.last_messages[0]["content"])
        self.assertIn("Previous analysis notes:", rendered)

    def test_chat_eval_prompt_can_use_assistant_prefix_mode(self) -> None:
        sample = {
            "problem": "Return x + 1.",
            "candidate_code": "def f(x):\n    return x + 1",
            "tests": ["assert f(1) == 2"],
            "language": "python",
        }
        prior_step = "static_logic_check: The function returns x + 1 directly.\n"
        tokenizer = self._FakeChatTokenizer()

        rendered = build_chat_eval_prompt(
            tokenizer,
            sample,
            "Correctness Verification",
            partial_response=prior_step,
            force_final=False,
            step_context_mode="assistant_prefix",
            show_tests_in_prompt=False,
        )

        self.assertEqual(len(tokenizer.last_messages), 1)
        self.assertIn("static_logic_check: The function returns x + 1 directly.", rendered)
        self.assertIn("inserted directly as your prior thinking history", tokenizer.last_messages[0]["content"])

    def test_chat_eval_prompt_appends_final_review_prefix_to_active_assistant_turn(self) -> None:
        sample = {
            "problem": "Return x + 1.",
            "candidate_code": "def f(x):\n    return x + 1",
            "tests": ["assert f(1) == 2"],
            "language": "python",
        }
        tokenizer = self._FakeChatTokenizer()

        rendered = build_chat_eval_prompt(
            tokenizer,
            sample,
            "Correctness Verification",
            force_final=True,
            final_only=True,
            step_context_mode="instruction_context",
            show_tests_in_prompt=False,
            enable_thinking=False,
        )

        self.assertEqual(len(tokenizer.last_messages), 1)
        self.assertEqual(tokenizer.last_kwargs["enable_thinking"], False)
        self.assertTrue(rendered.rstrip().endswith('<review>\n{"axiom_grade":'))
        self.assertEqual(rendered.count("[assistant]"), 1)

    def test_stepwise_eval_final_prompt_shows_review_format(self) -> None:
        sample = {
            "problem": "Return x + 1.",
            "candidate_code": "def f(x):\n    return x + 1",
            "tests": ["assert f(1) == 2"],
            "language": "python",
        }
        prior_step = "static_logic_check: The function returns x + 1 directly.\n"
        prompt = prompt_for_dimension(
            sample,
            "Correctness Verification",
            partial_response=prior_step,
            force_final=True,
        )

        self.assertIn("This is the final scoring turn", prompt)
        self.assertIn("Output exactly one <review> JSON block", prompt)
        self.assertIn("This is the final turn of a multi-step code review", prompt)
        self.assertIn("reconcile supported previous reasoning evidence", prompt)
        self.assertIn("cannot silently contradict it", prompt)
        self.assertIn('"axiom_grade"', prompt)
        self.assertIn("You don't have to analyze code by yourself.", prompt)
        self.assertIn("Do not continue the numbered analysis notes.", prompt)
        self.assertIn("Start your very first output token with <review>", prompt)
        self.assertIn("static_logic_check: The function returns x + 1 directly.", prompt)
        response_tail = prompt.split("@@ Response", 1)[1]
        self.assertNotIn(prior_step.strip(), response_tail)
        self.assertEqual(response_tail.strip(), "")

    def test_stepwise_eval_final_prompt_without_prior_steps_adds_no_prior_hint(self) -> None:
        sample = {
            "problem": "Return x + 1.",
            "candidate_code": "def f(x):\n    return x + 1",
            "tests": ["assert f(1) == 2"],
            "language": "python",
        }
        prompt = prompt_for_dimension(
            sample,
            "Correctness Verification",
            partial_response="",
            force_final=True,
        )

        self.assertNotIn("This is the final turn of a multi-step code review", prompt)
        self.assertNotIn("reconcile supported previous reasoning evidence", prompt)
        self.assertNotIn("You don't have to analyze code by yourself.", prompt)
        self.assertNotIn("Do not continue the numbered analysis notes.", prompt)

    def test_review_training_prompt_matches_response_shape(self) -> None:
        instruction = "Scoring target: assess candidate code correctness.\n\nTask description:\nReturn x + 1."

        final_prompt = review_prompt_for_response(instruction, "<review>\n{\"axiom_grade\": 5}\n</review>")
        step_prompt = review_prompt_for_response(instruction, "trace_requirement: Check return value.")

        self.assertIn("This is the final turn of a multi-step code review", final_prompt)
        self.assertIn("Output must be exactly one JSON object wrapped in <review> tags", final_prompt)
        self.assertIn("reconcile supported previous reasoning evidence", final_prompt)
        self.assertIn("cannot silently contradict it", final_prompt)
        self.assertIn("evidence_type", final_prompt)
        self.assertNotIn('"score"', final_prompt)
        self.assertNotIn('"verdict"', final_prompt)
        self.assertNotIn("stepwise evidence-and-review training", final_prompt)
        self.assertIn("This training response may contain intermediate native reasoning", step_prompt)
        self.assertIn("native reasoning", step_prompt)
        self.assertIn("Intermediate reasoning format", step_prompt)
        self.assertNotIn('"step_type"', step_prompt)
        self.assertIn("Final review format", step_prompt)
        self.assertNotIn("as long as needed", step_prompt)
        self.assertIn("reconcile supported reasoning evidence", step_prompt)
        self.assertIn("minor tweaking", final_prompt)
        self.assertIn("major refactoring", final_prompt)
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

    def test_shared_prompt_contract_does_not_export_codegen_prompts(self) -> None:
        import shared.prompt_contract as prompt_contract

        for name in [
            "DEEPSEEK_PROMPT",
            "DEEPSEEK_LCB_PROMPT",
            "QWEN_DIRECT_PROMPT",
            "QWEN_STEP_PROMPT",
        ]:
            self.assertFalse(hasattr(prompt_contract, name), name)


if __name__ == "__main__":
    unittest.main()

"""Smoke tests verifying that shared prompt contract is consistently
imported by data_collection and model_training wrapper modules, and that
chat-template-based eval prompts are well-formed.
"""

import unittest

from shared.prompt_contract import (
    AXIOM_REFINEMENT_SCALE,
    REVIEW_EVIDENCE_RULES,
    REVIEW_FINAL_CONSISTENCY_RULE,
    REVIEW_STEP_FORMAT_SECTION,
    REVIEW_FINAL_FORMAT_SECTION,
    FINAL_REVIEW_PREFILL,
    build_review_instruction,
    build_review_instruction_from_sample,
    build_review_user_content,
    build_review_prompt,
    prompt_to_chat_messages,
)


SAMPLE = {
    "problem": "Return x + 1.",
    "candidate_code": "def f(x):\n    return x + 1",
    "tests": ["assert f(1) == 2"],
    "language": "python",
}


class SharedConstantsTest(unittest.TestCase):
    """Shared constants are importable and non-empty."""

    def test_axiom_scale_present(self):
        self.assertIn("5/5", AXIOM_REFINEMENT_SCALE)

    def test_evidence_rules_present(self):
        self.assertIn("Evidence rules", REVIEW_EVIDENCE_RULES)

    def test_final_consistency_rule_present(self):
        self.assertIn("reconcile", REVIEW_FINAL_CONSISTENCY_RULE)

    def test_format_sections_present(self):
        self.assertIn("reasoning", REVIEW_STEP_FORMAT_SECTION)
        self.assertIn("<review>", REVIEW_FINAL_FORMAT_SECTION)

    def test_final_review_prefill(self):
        self.assertTrue(FINAL_REVIEW_PREFILL.startswith("<review>"))
        self.assertIn("axiom_grade", FINAL_REVIEW_PREFILL)


class ReExportAlignmentTest(unittest.TestCase):
    """Wrapper modules re-export the shared canonical constants."""

    def test_prompt_sft_reexports_axiom_scale(self):
        from mcts_math.prompts.prompt_sft import AXIOM_REFINEMENT_SCALE as sft_scale
        self.assertEqual(sft_scale, AXIOM_REFINEMENT_SCALE)

    def test_prompt_sft_reexports_evidence_rules(self):
        from mcts_math.prompts.prompt_sft import REVIEW_EVIDENCE_RULES as sft_rules
        self.assertEqual(sft_rules, REVIEW_EVIDENCE_RULES)

    def test_prompt_sft_keeps_legacy_raw_field_template(self):
        from mcts_math.prompts.prompt_sft import QWEN_REVIEW_STEP_PROMPT
        self.assertIn("{question}", QWEN_REVIEW_STEP_PROMPT)
        self.assertIn("{candidate_code}", QWEN_REVIEW_STEP_PROMPT)

    def test_prompt_template_reexports_axiom_scale(self):
        try:
            from magicoder.prompt_template import AXIOM_REFINEMENT_SCALE as pt_scale
        except ImportError:
            self.skipTest("magicoder deps not installed")
        self.assertEqual(pt_scale, AXIOM_REFINEMENT_SCALE)

    def test_prompt_template_keeps_legacy_response_template(self):
        try:
            from magicoder.prompt_template import QWEN_REVIEW_STEP_PROMPT
        except ImportError:
            self.skipTest("magicoder deps not installed")
        self.assertIn("{response}", QWEN_REVIEW_STEP_PROMPT)
        self.assertIn("{instruction}", QWEN_REVIEW_STEP_PROMPT)


class BuildInstructionTest(unittest.TestCase):
    """Unified instruction builder produces consistent output."""

    def test_basic_output_structure(self):
        text = build_review_instruction("Return x+1", "def f(x): return x+1")
        self.assertIn("Scoring target:", text)
        self.assertIn("Task description:", text)
        self.assertIn("Candidate code:", text)
        self.assertIn("Available tests:", text)

    def test_from_sample(self):
        text = build_review_instruction_from_sample(SAMPLE, "Correctness Verification")
        self.assertIn("Return x + 1.", text)
        self.assertIn("def f(x):", text)
        self.assertIn("No tests are available to the reviewer.", text)

    def test_from_sample_with_tests(self):
        text = build_review_instruction_from_sample(SAMPLE, "Correctness Verification", show_tests_in_prompt=True)
        self.assertIn("assert f(1) == 2", text)

    def test_matches_preprocess_build_instruction(self):
        try:
            from magicoder.preprocess_review_mcts_data import build_instruction
        except ImportError:
            self.skipTest("magicoder deps not installed")
        preprocess_text = build_instruction(SAMPLE, "Correctness Verification")
        shared_text = build_review_instruction_from_sample(SAMPLE, "Correctness Verification")
        self.assertEqual(preprocess_text, shared_text)


class BuildUserContentTest(unittest.TestCase):
    """User content builder matches training preprocessing."""

    def test_user_content_has_preamble(self):
        instruction = build_review_instruction_from_sample(SAMPLE, "Correctness Verification")
        user_content = build_review_user_content(instruction)
        self.assertIn("code scoring model", user_content)
        self.assertIn("<review>", user_content)
        self.assertIn(instruction.strip(), user_content)

    def test_matches_preprocess_qwen_review_user_content(self):
        try:
            from magicoder.preprocess_review_mcts_data import build_instruction, qwen_review_user_content
        except ImportError:
            self.skipTest("magicoder deps not installed")
        instruction = build_instruction(SAMPLE, "Correctness Verification")
        preprocess_content = qwen_review_user_content(instruction)
        shared_content = build_review_user_content(instruction)
        self.assertEqual(preprocess_content, shared_content)


class BuildReviewPromptTest(unittest.TestCase):
    """build_review_prompt produces valid prompt strings."""

    def test_step_prompt_has_response_marker(self):
        instruction = build_review_instruction_from_sample(SAMPLE, "Correctness Verification")
        prompt = build_review_prompt(instruction, force_final=False)
        self.assertIn("@@ Response", prompt)
        self.assertNotIn(FINAL_REVIEW_PREFILL, prompt)

    def test_final_prompt_has_prefill(self):
        instruction = build_review_instruction_from_sample(SAMPLE, "Correctness Verification")
        prompt = build_review_prompt(instruction, force_final=True)
        self.assertIn("@@ Response", prompt)
        self.assertIn(FINAL_REVIEW_PREFILL, prompt)

    def test_partial_solution_appears_in_prompt(self):
        instruction = build_review_instruction_from_sample(SAMPLE, "Correctness Verification")
        partial = "<step>check: x+1 is correct</step>"
        prompt = build_review_prompt(instruction, partial_solution=partial, force_final=True)
        self.assertIn(partial, prompt)
        self.assertIn(FINAL_REVIEW_PREFILL, prompt)


class PromptToChatMessagesTest(unittest.TestCase):
    """prompt_to_chat_messages splits correctly at @@ Response."""

    def test_no_response_section(self):
        msgs = prompt_to_chat_messages("Just a user message")
        self.assertEqual(len(msgs), 1)
        self.assertEqual(msgs[0]["role"], "user")

    def test_empty_response(self):
        msgs = prompt_to_chat_messages("instruction\n@@ Response\n")
        self.assertEqual(len(msgs), 1)

    def test_with_response_prefix(self):
        msgs = prompt_to_chat_messages("instruction\n@@ Response\n<step>hello</step>")
        self.assertEqual(len(msgs), 2)
        self.assertEqual(msgs[0]["role"], "user")
        self.assertEqual(msgs[1]["role"], "assistant")
        self.assertIn("<step>hello</step>", msgs[1]["content"])

    def test_strips_think_suffix(self):
        msgs = prompt_to_chat_messages("some prompt\n\n/think")
        self.assertEqual(len(msgs), 1)
        self.assertNotIn("/think", msgs[0]["content"])


if __name__ == "__main__":
    unittest.main()

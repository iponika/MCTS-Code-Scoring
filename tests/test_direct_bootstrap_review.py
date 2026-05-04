import unittest
from types import SimpleNamespace
from unittest.mock import patch

from omegaconf import OmegaConf

import direct_bootstrap_review
from mcts_math.config import BaseConfig


class _FakeEngine:
    def __init__(self, output_text: str) -> None:
        self.output_text = output_text
        self.last_prompts = None
        self.last_n = None
        self.last_best_of = None

    def generate(self, prompts, sampling_params=None):
        self.last_prompts = list(prompts)
        self.last_n = getattr(sampling_params, "n", None)
        self.last_best_of = getattr(sampling_params, "best_of", None)
        return [
            SimpleNamespace(outputs=[SimpleNamespace(text=self.output_text)])
            for _ in prompts
        ]


class DirectBootstrapReviewTest(unittest.TestCase):
    def test_review_only_repeats_duplicate_prompts_not_greedy_n(self) -> None:
        sample = {
            "question": "Return x + 1.",
            "candidate_code": "def f(x):\n    return x + 1\n",
            "code_language": "python",
            "tests_for_prompt": "No tests are available to the reviewer.",
        }
        args = SimpleNamespace(repeats=2, dimension="Correctness Verification")
        config = OmegaConf.structured(BaseConfig)
        sampling_params = SimpleNamespace(n=None, best_of=None, stop=None, temperature=0.7, top_p=0.9)
        engine = _FakeEngine('"axiom_grade": 4, "overall_assessment": "ok", "objective_evidence": [], "risk_level": "low"}')

        with (
            patch.object(direct_bootstrap_review, "maybe_apply_chat_template", side_effect=lambda prompts, *_: prompts),
            patch.object(
                direct_bootstrap_review,
                "evaluated_candidate",
                side_effect=lambda index, text, sample, dimension: {
                    "candidate_index": index,
                    "text": text,
                    "reward": 0.0,
                    "reward_details": {},
                    "predicted_axiom_grade": None,
                },
            ),
        ):
            results = direct_bootstrap_review.generate_review_only([sample], args, config, engine, sampling_params)

        self.assertEqual(engine.last_n, 1)
        self.assertEqual(engine.last_best_of, 1)
        self.assertEqual(len(engine.last_prompts), 2)
        self.assertEqual(len(results), 1)
        self.assertEqual(len(results[0][1]), 2)
        self.assertEqual([item["candidate_index"] for item in results[0][1]], [0, 1])

    def test_review_only_final_prompt_is_freeform_by_default(self) -> None:
        sample = {
            "question": "Return x + 1.",
            "candidate_code": "def f(x):\n    return x + 1\n",
            "code_language": "python",
            "tests_for_prompt": "No tests are available to the reviewer.",
        }
        args = SimpleNamespace(repeats=1, dimension="Correctness Verification")
        config = OmegaConf.structured(BaseConfig)
        config.review_native_thinking_steps = True
        sampling_params = SimpleNamespace(n=None, best_of=None, stop=None, temperature=0.7, top_p=0.9)
        engine = _FakeEngine('<think>\nreason\n</think>\n<review>{"axiom_grade": 4}</review>')
        captured_texts = []

        with (
            patch.object(direct_bootstrap_review, "maybe_apply_chat_template", side_effect=lambda prompts, *_: prompts),
            patch.object(
                direct_bootstrap_review,
                "evaluated_candidate",
                side_effect=lambda index, text, sample, dimension: (
                    captured_texts.append(text)
                    or {
                        "candidate_index": index,
                        "text": text,
                        "reward": 0.0,
                        "reward_details": {},
                        "predicted_axiom_grade": None,
                    }
                ),
            ),
        ):
            direct_bootstrap_review.generate_review_only([sample], args, config, engine, sampling_params)

        self.assertEqual(len(engine.last_prompts), 1)
        self.assertNotIn(direct_bootstrap_review.FINAL_REVIEW_PREFILL, engine.last_prompts[0])
        self.assertFalse(engine.last_prompts[0].rstrip().endswith("/no_think"))
        self.assertIn("You may reason before the final labeled answer", engine.last_prompts[0])
        self.assertIn("The final labeled answer begins at the last complete <review> block", engine.last_prompts[0])
        self.assertNotIn("The very first non-whitespace characters of your answer must be <review>.", engine.last_prompts[0])
        self.assertEqual(captured_texts, ['<think>\nreason\n</think>\n<review>{"axiom_grade": 4}</review>'])

    def test_review_only_can_explicitly_use_prefill_anchor(self) -> None:
        sample = {
            "question": "Return x + 1.",
            "candidate_code": "def f(x):\n    return x + 1\n",
            "code_language": "python",
            "tests_for_prompt": "No tests are available to the reviewer.",
        }
        args = SimpleNamespace(repeats=1, dimension="Correctness Verification", freeform_final_review=False)
        config = OmegaConf.structured(BaseConfig)
        config.review_native_thinking_steps = True
        sampling_params = SimpleNamespace(n=None, best_of=None, stop=None, temperature=0.7, top_p=0.9)
        engine = _FakeEngine('"axiom_grade": 4, "summary": "ok"}\n</review>')

        with (
            patch.object(direct_bootstrap_review, "maybe_apply_chat_template", side_effect=lambda prompts, *_: prompts),
            patch.object(
                direct_bootstrap_review,
                "evaluated_candidate",
                side_effect=lambda index, text, sample, dimension: {
                    "candidate_index": index,
                    "text": text,
                    "reward": 0.0,
                    "reward_details": {},
                    "predicted_axiom_grade": None,
                },
            ),
        ):
            direct_bootstrap_review.generate_review_only([sample], args, config, engine, sampling_params)

        self.assertIn(direct_bootstrap_review.FINAL_REVIEW_PREFILL, engine.last_prompts[0])
        self.assertTrue(engine.last_prompts[0].rstrip().endswith("/no_think"))

    def test_normalize_review_text_keeps_only_last_review_block(self) -> None:
        text = (
            "First I compare the requirement to the code.\n"
            "<review>{\"axiom_grade\": 1}</review>\n"
            "After reconsidering, the final answer is:\n"
            "<review>{\"axiom_grade\": 4, \"score\": 80}</review>"
        )
        normalized = direct_bootstrap_review.normalize_review_text(text)
        self.assertEqual(normalized, '<review>\n{"axiom_grade": 4, "score": 80}\n</review>')

    def test_merge_final_review_text_avoids_double_axiom_prefix_for_full_json_object(self) -> None:
        text = '{"axiom_grade": 0, "functional_correctness": false}\n</review>'

        merged = direct_bootstrap_review.merge_final_review_text(text, anchored=True)

        self.assertEqual(merged, '<review>\n{"axiom_grade": 0, "functional_correctness": false}\n</review>')

    def test_merge_final_review_text_restores_axiom_prefix_for_closing_only_suffix(self) -> None:
        text = '2, "functional_correctness": false, "repair_effort": "minor_functional"}\n</review>'

        merged = direct_bootstrap_review.merge_final_review_text(text, anchored=True)

        self.assertTrue(merged.startswith(direct_bootstrap_review.FINAL_REVIEW_PREFILL))


if __name__ == "__main__":
    unittest.main()

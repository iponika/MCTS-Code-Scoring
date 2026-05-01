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


if __name__ == "__main__":
    unittest.main()

# Review Final Prompt Backup Before Rewrite

Date: 2026-05-02

This file preserves the final-review prompt fragments before the May 2 prompt rewrite.

## data_collection/mcts_math/prompts/prompt_sft.py

```text
REVIEW_FINAL_FORMAT_SECTION = """Structured final review format:
<review>
{{"axiom_grade": <0-5 integer>, "score": <0-100 number>, "verdict": "accept|minor_issue|major_issue", "functional_correctness": true, "repair_effort": "none|minor_quality|major_quality|minor_functional|major_functional|rewrite", "evidence_type": "provided_test_failure|deduced_counterexample|static_logic_contradiction|uncertain", "summary": "...", "evidence": ["...", "..."]}}
</review>"""

QWEN_REVIEW_FINAL_PROMPT = """You are a code scoring model for functional correctness.
@@ Instruction
Current generation mode: final AXIOM scoring.
Output exactly one compact JSON object wrapped in <review> tags. Do not output <step> blocks or prose outside <review>.
The very first non-whitespace characters of your answer must be <review>.
After the opening tag, continue immediately with a JSON object, not a natural-language sentence.
The response is already inside the final JSON object. The first JSON key must be "axiom_grade".
The next value after "axiom_grade": must be one integer in 0-5, followed immediately by a comma.
Do not output <think>, <step>, markdown fences, or explanatory prose in this final turn.
Treat completed previous steps as fixed evidence context. Use them if supported by the task/code/tests; ignore unsupported or repeated claims.
..."""
```

## model_training/src/magicoder/prompt_template.py

```text
QWEN_REVIEW_STEP_PROMPT final review format:
<review>
{{"axiom_grade": <0-5 integer>, "score": <0-100 number>, "verdict": "accept|minor_issue|major_issue", "functional_correctness": true, "repair_effort": "none|minor_quality|major_quality|minor_functional|major_functional|rewrite", "summary": "...", "evidence": ["...", "..."]}}
</review>

QWEN_REVIEW_FINAL_ONLY_PROMPT required keys:
axiom_grade, score, verdict, functional_correctness, repair_effort, summary, evidence
```

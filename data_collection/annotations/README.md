# Annotation Files

## ComplexCodeEval AXIOM Relabels

`complexcodeeval_axiom_30.jsonl` contains the first manual AXIOM relabel pass.

`complexcodeeval_highscore_axiom_relabels.jsonl` contains a focused pass over original high-score ComplexCodeEval samples. It keeps the legacy-compatible top-level `axiom_grade` and adds a structured `axiom_label` object for downstream parsing.

Recommended parser fields:

- `source_line`: 1-based line number in `test-E-responded.jsonl` and `test-0-human.jsonl`.
- `original_human_score`: original ComplexCodeEval human score.
- `axiom_label.grade`: AXIOM grade after pass/fail-first relabeling.
- `axiom_label.functional_correctness`: whether the sample has no verified functional defect.
- `axiom_label.modified`: whether the AXIOM label differs materially from the original high-score interpretation.
- `axiom_label.evidence_type`, `axiom_label.summary`, `axiom_label.evidence`: audit trail for manual review.

## Excluded Samples

`excluded_samples.jsonl` is the canonical registry for samples that should be skipped by data loading because their labels or source content are currently unsuitable for training/evaluation.

Records are JSONL with `status=active` for enabled exclusions. The loader matches stable provenance fields such as `dataset_family`, `source`, `subset`, `source_dataset`, `original_dataset_index`, and `dataset_index`; `prepared_seed_line_index` is included when the current seed row is known, but should not be the only identifier.

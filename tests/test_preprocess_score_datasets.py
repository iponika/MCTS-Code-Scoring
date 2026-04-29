import json
import tempfile
import unittest
from pathlib import Path

from magicoder.preprocess_score_datasets import load_axiom_items


class PreprocessScoreDatasetsTest(unittest.TestCase):
    def test_load_axiom_items_can_drop_grade_zero(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            axiom_dir = Path(tmp_dir) / "axiombench"
            axiom_dir.mkdir()
            path = axiom_dir / "apps.jsonl"
            rows = [
                {"score": 0, "inst": "Return one.", "code": "def f(): return 0", "lang": "python"},
                {"score": 3, "inst": "Return one.", "code": "def f(): return 1", "lang": "python"},
            ]
            path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

            items = list(load_axiom_items(Path(tmp_dir), train_lm_exact=False, limit=0, drop_grade_zero=True))

        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["target_axiom_grade"], 3)


if __name__ == "__main__":
    unittest.main()

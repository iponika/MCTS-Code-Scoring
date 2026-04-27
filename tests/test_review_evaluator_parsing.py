import unittest

from magicoder.review_evaluator import parse_final_review


class ReviewEvaluatorParsingTest(unittest.TestCase):
    def test_parse_final_review_uses_last_review_and_ignores_extra_brace(self) -> None:
        text = (
            '<review>{"axiom_grade": 5, "score": 100}</review>\n'
            '<review>{"axiom_grade": 2, "score": 40, "evidence": ["x"]}}</review>'
        )

        parsed = parse_final_review(text)

        self.assertTrue(parsed["ok"])
        self.assertEqual(parsed["parsed"]["axiom_grade"], 2)
        self.assertTrue(parsed.get("recovered"))

    def test_parse_final_review_escapes_control_chars_inside_strings(self) -> None:
        text = '<review>{"axiom_grade": 3, "score": 60, "evidence": ["line one\nline two"]}</review>'

        parsed = parse_final_review(text)

        self.assertTrue(parsed["ok"])
        self.assertEqual(parsed["parsed"]["evidence"], ["line one\nline two"])
        self.assertTrue(parsed.get("recovered"))

    def test_parse_final_review_recovers_minimal_grade_from_truncated_json(self) -> None:
        text = '<review>{"axiom_grade": 4, "score": 80, "evidence": ["unterminated</review>'

        parsed = parse_final_review(text)

        self.assertTrue(parsed["ok"])
        self.assertEqual(parsed["parsed"]["axiom_grade"], 4)
        self.assertEqual(parsed["parsed"]["score"], 80.0)
        self.assertEqual(parsed.get("recovery_method"), "grade_fallback")


if __name__ == "__main__":
    unittest.main()

import unittest

from magicoder.review_evaluator import extract_reasoning_artifacts, parse_final_review


class ReviewEvaluatorParsingTest(unittest.TestCase):
    def test_extract_reasoning_artifacts_from_text_think_block(self) -> None:
        artifacts = extract_reasoning_artifacts(
            "<think>\ncheck the boundary case\n</think>\n<review>{\"axiom_grade\":2}</review>"
        )

        self.assertEqual(artifacts["reasoning"], "check the boundary case")
        self.assertEqual(artifacts["content"], '<review>{"axiom_grade":2}</review>')
        self.assertEqual(artifacts["reasoning_source"], "text_think_block")

    def test_extract_reasoning_artifacts_prefers_structured_reasoning(self) -> None:
        artifacts = extract_reasoning_artifacts(
            "<review>{\"axiom_grade\":5}</review>",
            structured_reasoning="first verify the concrete requirement trace",
        )

        self.assertEqual(artifacts["reasoning"], "first verify the concrete requirement trace")
        self.assertEqual(artifacts["content"], '<review>{"axiom_grade":5}</review>')
        self.assertEqual(artifacts["reasoning_source"], "structured_field")

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

import os
import sys
import unittest


TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(TESTS_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from core.option_reasoning import (
    program_uses_unsupported_closest_estimate_rounding,
    select_option_from_values,
)


class TestOptionReasoning(unittest.TestCase):
    def test_closest_estimate_prefers_numeric_value_over_conflicting_bare_letter(self):
        question = (
            "Jimmy and Kima are going on a trip. They will drive for three days. "
            "The first day they will drive 182 miles. The second day they will drive 439 miles. "
            "The third day they will drive 217 miles. Which expression is the closest estimate "
            "of how many miles Jimmy and Kima will drive on their trip?"
        )
        options = [
            {"key": "A", "label": "150 + 400 + 200"},
            {"key": "B", "label": "200 + 400 + 200"},
            {"key": "C", "label": "200 + 450 + 200"},
            {"key": "D", "label": "200 + 500 + 200"},
        ]

        resolved = select_option_from_values(
            ["838", "The answer is D"],
            options,
            question_text=question,
        )

        self.assertIsNotNone(resolved)
        self.assertEqual(resolved["key"], "C")
        self.assertEqual(resolved["match_type"], "closest-numeric")

    def test_closest_estimate_falls_back_to_explicit_option_when_no_numeric_value_exists(self):
        question = "Which expression is the closest estimate?"
        options = [
            {"key": "A", "label": "100"},
            {"key": "B", "label": "200"},
            {"key": "C", "label": "300"},
            {"key": "D", "label": "400"},
        ]

        resolved = select_option_from_values(
            ["Final Answer: D"],
            options,
            question_text=question,
        )

        self.assertIsNotNone(resolved)
        self.assertEqual(resolved["key"], "D")
        self.assertEqual(resolved["match_type"], "explicit-option")

    def test_program_rounding_shortcut_is_detected_for_closest_estimate(self):
        question = (
            "Jimmy and Kima are going on a trip. Which expression is the closest estimate "
            "of how many miles they will drive?"
        )
        program = (
            "from sympy import *\n"
            "day1_rounded = round(182 / 100) * 100\n"
            "day2_rounded = round(439 / 100) * 100\n"
            "print(day1_rounded + day2_rounded)\n"
        )

        self.assertTrue(program_uses_unsupported_closest_estimate_rounding(question, program))

    def test_program_rounding_shortcut_is_allowed_when_question_explicitly_requests_rounding(self):
        question = (
            "Round each day's miles to the nearest hundred, then choose the closest estimate "
            "for the total."
        )
        program = (
            "from sympy import *\n"
            "day1_rounded = round(182 / 100) * 100\n"
            "day2_rounded = round(439 / 100) * 100\n"
            "print(day1_rounded + day2_rounded)\n"
        )

        self.assertFalse(program_uses_unsupported_closest_estimate_rounding(question, program))

    def test_resolved_numeric_comparison_annotation_maps_to_the_correct_option(self):
        question = (
            "Jimmy and Kima are going on a trip. Which expression is the closest estimate "
            "of how many miles they will drive?"
        )
        options = [
            {"key": "A", "label": "150 + 400 + 200"},
            {"key": "B", "label": "200 + 400 + 200"},
            {"key": "C", "label": "200 + 450 + 200"},
            {"key": "D", "label": "200 + 500 + 200"},
        ]

        resolved = select_option_from_values(
            ["Resolved option by numeric comparison: C. 200 + 450 + 200"],
            options,
            question_text=question,
        )

        self.assertIsNotNone(resolved)
        self.assertEqual(resolved["key"], "C")


if __name__ == "__main__":
    unittest.main()

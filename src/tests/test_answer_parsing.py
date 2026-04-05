import os
import sys
import unittest


TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(TESTS_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from core.answer_parsing import (
    extract_final_answer_option_letter,
    extract_numeric_answer,
    extract_option_letter,
    extract_preferred_answer,
    extract_tagged_answer,
)


class TestAnswerParsing(unittest.TestCase):
    def test_extract_numeric_answer_supports_legacy_and_clean_formats(self):
        self.assertEqual(extract_numeric_answer("#### 18"), "18")
        self.assertEqual(extract_numeric_answer("Final Answer: 18"), "18")
        self.assertEqual(extract_numeric_answer("Work here\nThe answer is 18"), "18")

    def test_extract_option_letter_supports_final_answer_prefix(self):
        self.assertEqual(extract_option_letter("Final Answer: B"), "B")
        self.assertEqual(extract_option_letter("The answer is C"), "C")
        self.assertEqual(extract_option_letter("Answer: D"), "D")

    def test_extract_final_answer_option_letter_prefers_explicit_final_answer_tag(self):
        text = "We compute x - y = -48.\nFinal Answer: D"
        self.assertEqual(extract_final_answer_option_letter(text), "D")

    def test_extract_option_letter_ignores_algebraic_coefficient_lines(self):
        text = "A + B + C = 3\n2A + B = 2\n9A + 3B + C = 15"
        self.assertIsNone(extract_option_letter(text))

    def test_extract_tagged_answer_uses_last_explicit_answer(self):
        text = "Wolfram_Alpha response: The answer is 3 when z = 1 or z = -1.\nFinal Answer: 5"
        self.assertEqual(extract_tagged_answer(text), "5")

    def test_extract_preferred_answer_uses_rhs_of_labeled_output(self):
        self.assertEqual(extract_preferred_answer("g(f(-1)): 2"), "2")

    def test_extract_preferred_answer_strips_markdown_wrappers(self):
        self.assertEqual(extract_preferred_answer("**Final answer: 0**"), "0")

    def test_extract_numeric_answer_accepts_number_words(self):
        self.assertEqual(extract_numeric_answer("Final Answer: two"), "2")

    def test_extract_numeric_answer_preserves_negative_sign(self):
        self.assertEqual(extract_numeric_answer("-48"), "-48")


if __name__ == "__main__":
    unittest.main()

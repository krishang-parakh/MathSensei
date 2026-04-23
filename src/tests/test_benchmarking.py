import os
import sys
import unittest
from unittest.mock import mock_open, patch


TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(TESTS_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from presentation.benchmarking import evaluate_record, summarize_accuracy, update_aggregate_benchmark


class TestBenchmarking(unittest.TestCase):
    def test_gsm_numeric_evaluation_is_lenient_on_formatting(self):
        record = {
            "dataset": "GSM",
            "final_answer": "18.0",
            "ground_truth_solution": "Work here\n#### 18",
            "gold_answer": "#### 18",
        }

        evaluation = evaluate_record(record)

        self.assertEqual(evaluation["evaluation_status"], "evaluated")
        self.assertTrue(evaluation["is_correct"])
        self.assertEqual(evaluation["evaluation_method"], "gsm-numeric")

    def test_gsm_numeric_evaluation_accepts_final_answer_label(self):
        record = {
            "dataset": "GSM",
            "final_generated_solution": "We compute the value.\nFinal Answer: 22",
            "ground_truth_solution": "#### 22",
            "gold_answer": "#### 22",
        }

        evaluation = evaluate_record(record)

        self.assertEqual(evaluation["evaluation_status"], "evaluated")
        self.assertTrue(evaluation["is_correct"])

    def test_gsm_numeric_evaluation_extracts_number_from_sentence_answer(self):
        record = {
            "dataset": "GSM",
            "final_answer": "She must hike 60 miles more per day.",
            "ground_truth_solution": "#### 60",
            "gold_answer": "60",
        }

        evaluation = evaluate_record(record)

        self.assertEqual(evaluation["evaluation_status"], "evaluated")
        self.assertTrue(evaluation["is_correct"])
        self.assertEqual(evaluation["predicted_answer"], "She must hike 60 miles more per day")

    def test_gsm_numeric_evaluation_handles_markdown_wrapped_numeric_answer(self):
        record = {
            "dataset": "GSM",
            "final_answer": "0**",
            "ground_truth_solution": "#### 60",
            "gold_answer": "60",
        }

        evaluation = evaluate_record(record)

        self.assertEqual(evaluation["evaluation_status"], "evaluated")
        self.assertFalse(evaluation["is_correct"])

    def test_explicit_final_answer_takes_precedence_over_solution_text(self):
        record = {
            "dataset": "GSM",
            "final_answer": "22",
            "final_generated_solution": "We compute the value.\nFinal Answer: 99",
            "ground_truth_solution": "#### 22",
            "gold_answer": "#### 22",
        }

        evaluation = evaluate_record(record)

        self.assertEqual(evaluation["evaluation_status"], "evaluated")
        self.assertTrue(evaluation["is_correct"])
        self.assertEqual(evaluation["predicted_answer"], "22")

    def test_option_dataset_resolves_and_matches_option_values(self):
        record = {
            "dataset": "MMLU",
            "final_answer": "B. Beta",
            "correct_option": "B",
            "ground_truth_solution": "B",
            "options": [
                {"key": "A", "label": "Alpha"},
                {"key": "B", "label": "Beta"},
                {"key": "C", "label": "Gamma"},
                {"key": "D", "label": "Delta"},
            ],
        }

        evaluation = evaluate_record(record)

        self.assertEqual(evaluation["evaluation_status"], "evaluated")
        self.assertTrue(evaluation["is_correct"])
        self.assertEqual(evaluation["predicted_answer"], "Beta")
        self.assertEqual(evaluation["evaluation_method"], "option-value")

    def test_option_dataset_accepts_final_answer_label(self):
        record = {
            "dataset": "MMLU",
            "final_generated_solution": "Reasoning here.\nFinal Answer: C",
            "correct_option": "C",
            "ground_truth_solution": "C",
            "options": [
                {"key": "A", "label": "Alpha"},
                {"key": "B", "label": "Beta"},
                {"key": "C", "label": "Gamma"},
                {"key": "D", "label": "Delta"},
            ],
        }

        evaluation = evaluate_record(record)

        self.assertEqual(evaluation["evaluation_status"], "evaluated")
        self.assertTrue(evaluation["is_correct"])

    def test_option_dataset_accepts_formatted_final_answer_line_with_letter_and_value(self):
        record = {
            "dataset": "MMLU",
            "final_generated_solution": "Reasoning here.\nFinal Answer: C. Gamma",
            "correct_option": "C",
            "ground_truth_solution": "C",
            "options": [
                {"key": "A", "label": "Alpha"},
                {"key": "B", "label": "Beta"},
                {"key": "C", "label": "Gamma"},
                {"key": "D", "label": "Delta"},
            ],
        }

        evaluation = evaluate_record(record)

        self.assertEqual(evaluation["evaluation_status"], "evaluated")
        self.assertTrue(evaluation["is_correct"])
        self.assertEqual(evaluation["predicted_answer"], "Gamma")

    def test_option_dataset_prefers_resolved_value_over_conflicting_bare_letter(self):
        record = {
            "dataset": "AQUA",
            "final_generated_solution": (
                "0.018N = 72\n\n"
                "N = 72 / 0.018 = 4000\n\n"
                "Thus, the total number of staplers produced that day is 4000.\n\n"
                "The answer is E."
            ),
            "correct_option": "A",
            "ground_truth_solution": "Answer : A",
            "options": [
                {"key": "A", "label": "4,000"},
                {"key": "B", "label": "4,200"},
                {"key": "C", "label": "4,500"},
                {"key": "D", "label": "4,800"},
                {"key": "E", "label": "5,000"},
            ],
        }

        evaluation = evaluate_record(record)

        self.assertEqual(evaluation["evaluation_status"], "evaluated")
        self.assertTrue(evaluation["is_correct"])
        self.assertEqual(evaluation["predicted_answer"], "4,000")

    def test_option_dataset_prefers_explicit_final_answer_tag_over_intermediate_values(self):
        record = {
            "dataset": "AQUA",
            "final_generated_solution": (
                "Step 4: Compute x - y.\n\n"
                "x - y = -23 - 25 = -48\n\n"
                "Alternatively, since x + y = 2, and x - y is what we need, the direct calculation confirms the value.\n\n"
                "Final Answer: D"
            ),
            "correct_option": "D",
            "ground_truth_solution": "Answer : D",
            "options": [
                {"key": "A", "label": "725"},
                {"key": "B", "label": "-2"},
                {"key": "C", "label": "2"},
                {"key": "D", "label": "-48"},
            ],
        }

        evaluation = evaluate_record(record)

        self.assertEqual(evaluation["evaluation_status"], "evaluated")
        self.assertTrue(evaluation["is_correct"])
        self.assertEqual(evaluation["predicted_answer"], "-48")

    def test_option_dataset_ignores_rejected_option_lines_when_solution_ends_with_answer_letter(self):
        record = {
            "dataset": "MMLU",
            "final_generated_solution": (
                "To find the number that makes 35 / ? = 7 true, solve for ?:\n"
                "? = 35 \u00f7 7 = 5\n\n"
                "Check options:\n"
                "A: 5 \u2014 matches the solution.\n"
                "B: 6 \u2014 not correct.\n"
                "C: 7 \u2014 not correct.\n"
                "D: 8 \u2014 not correct.\n\n"
                "The answer is A"
            ),
            "correct_option": "A",
            "ground_truth_solution": "A",
            "options": [
                {"key": "A", "label": "5"},
                {"key": "B", "label": "6"},
                {"key": "C", "label": "7"},
                {"key": "D", "label": "8"},
            ],
        }

        evaluation = evaluate_record(record)

        self.assertEqual(evaluation["evaluation_status"], "evaluated")
        self.assertTrue(evaluation["is_correct"])
        self.assertEqual(evaluation["predicted_answer"], "5")

    def test_option_dataset_ignores_trailing_option_list_after_explicit_answer_letter(self):
        record = {
            "dataset": "AQUA",
            "final_generated_solution": (
                "3000 = 100 * other side\n"
                "other side = 3000 / 100 = 30 ft\n\n"
                "Check options:\n"
                "A) 30 ft\n"
                "B) 20 ft\n"
                "C) 10 ft\n"
                "D) 50 ft\n"
                "E) 60 ft\n\n"
                "The answer is A"
            ),
            "correct_option": "A",
            "ground_truth_solution": "Answer : A",
            "options": [
                {"key": "A", "label": "30 feet"},
                {"key": "B", "label": "20 feet"},
                {"key": "C", "label": "10 feet"},
                {"key": "D", "label": "50 feet"},
                {"key": "E", "label": "60 feet"},
            ],
        }

        evaluation = evaluate_record(record)

        self.assertEqual(evaluation["evaluation_status"], "evaluated")
        self.assertTrue(evaluation["is_correct"])
        self.assertEqual(evaluation["predicted_answer"], "30 feet")

    def test_option_dataset_returns_value_and_display_from_the_same_resolution(self):
        record = {
            "dataset": "AQUA",
            "final_generated_solution": (
                "3000 = 100 * other side\n"
                "other side = 3000 / 100 = 30 ft\n\n"
                "Check options:\n"
                "A) 30 ft\n"
                "B) 20 ft\n"
                "C) 10 ft\n"
                "D) 50 ft\n"
                "E) 60 ft\n\n"
                "The answer is A"
            ),
            "correct_option": "A",
            "ground_truth_solution": "Answer : A",
            "options": [
                {"key": "A", "label": "30 feet"},
                {"key": "B", "label": "20 feet"},
                {"key": "C", "label": "10 feet"},
                {"key": "D", "label": "50 feet"},
                {"key": "E", "label": "60 feet"},
            ],
        }

        evaluation = evaluate_record(record)

        self.assertEqual(evaluation["predicted_answer"], "30 feet")
        self.assertEqual(evaluation["predicted_answer_display"], "A. 30 feet")
        self.assertEqual(evaluation["gold_answer"], "30 feet")
        self.assertEqual(evaluation["gold_answer_display"], "A. 30 feet")
        self.assertEqual(evaluation["predicted_answer_option"], "A")
        self.assertEqual(evaluation["gold_answer_option"], "A")

    def test_option_dataset_ignores_plain_option_line_that_survives_tail_window(self):
        record = {
            "dataset": "MMLU",
            "final_generated_solution": (
                "Solve 35 / z = 7\n\n"
                "Multiply both sides by z\n"
                "35 = 7z\n\n"
                "Divide by 7\n"
                "z = 5\n\n"
                "Check options:\n"
                "A: 5 matches the solution\n"
                "B: 6 not correct\n"
                "C: 7\n"
                "D: 8 not correct\n\n"
                "The answer is A"
            ),
            "correct_option": "A",
            "ground_truth_solution": "A",
            "options": [
                {"key": "A", "label": "5"},
                {"key": "B", "label": "6"},
                {"key": "C", "label": "7"},
                {"key": "D", "label": "8"},
            ],
        }

        evaluation = evaluate_record(record)

        self.assertEqual(evaluation["evaluation_status"], "evaluated")
        self.assertTrue(evaluation["is_correct"])
        self.assertEqual(evaluation["predicted_answer"], "5")

    def test_option_dataset_matches_precise_numeric_output_to_rounded_option_value(self):
        record = {
            "dataset": "MMLU",
            "final_answer": "Circumference: 5.34070751110265",
            "gold_answer": "D. 5.34 cm",
            "correct_option": "D",
            "ground_truth_solution": "D. 5.34 cm",
            "options": [
                {"key": "A", "label": "1.33 cm"},
                {"key": "B", "label": "1.70 cm"},
                {"key": "C", "label": "2.67 cm"},
                {"key": "D", "label": "5.34 cm"},
            ],
        }

        evaluation = evaluate_record(record)

        self.assertEqual(evaluation["evaluation_status"], "evaluated")
        self.assertTrue(evaluation["is_correct"])
        self.assertEqual(evaluation["predicted_answer"], "5.34 cm")
        self.assertEqual(evaluation["gold_answer"], "5.34 cm")

    def test_aqua_numeric_answer_matches_option_value(self):
        record = {
            "dataset": "AQUA",
            "final_answer": "13.6602540378444",
            "correct_option": "A",
            "ground_truth_solution": "Answer : A",
            "options": [
                {"key": "A", "label": "5(sqrt(3) + 1)"},
                {"key": "B", "label": "6(sqrt(3) + sqrt(2))"},
                {"key": "C", "label": "7(sqrt(3) - 1)"},
                {"key": "D", "label": "8(sqrt(3) - 2)"},
                {"key": "E", "label": "None of these"},
            ],
        }

        evaluation = evaluate_record(record)

        self.assertEqual(evaluation["evaluation_status"], "evaluated")
        self.assertTrue(evaluation["is_correct"])
        self.assertEqual(evaluation["gold_answer"], "5(sqrt(3) + 1)")

    def test_aqua_closest_estimate_question_maps_numeric_total_to_nearest_option(self):
        record = {
            "dataset": "AQUA",
            "problem": (
                "Jimmy and Kima are going on a trip. They will drive for three days. "
                "The first day they will drive 182 miles. The second day they will drive 439 miles. "
                "The third day they will drive 217 miles. Which expression is the closest estimate "
                "of how many miles Jimmy and Kima will drive on their trip?"
            ),
            "final_answer": "838",
            "correct_option": "C",
            "ground_truth_solution": "Answer : C",
            "options": [
                {"key": "A", "label": "150 + 400 + 200"},
                {"key": "B", "label": "200 + 400 + 200"},
                {"key": "C", "label": "200 + 450 + 200"},
                {"key": "D", "label": "200 + 500 + 200"},
            ],
        }

        evaluation = evaluate_record(record)

        self.assertEqual(evaluation["evaluation_status"], "evaluated")
        self.assertTrue(evaluation["is_correct"])
        self.assertEqual(evaluation["predicted_answer"], "200 + 450 + 200")

    def test_math_uses_normalized_answer_matching(self):
        record = {
            "dataset": "MATH",
            "final_answer": "\\frac{1}{2}",
            "ground_truth_solution": "Therefore the answer is \\boxed{\\frac{1}{2}}",
        }

        evaluation = evaluate_record(record)

        self.assertEqual(evaluation["evaluation_status"], "evaluated")
        self.assertTrue(evaluation["is_correct"])

    def test_math_evaluation_accepts_plus_minus_form_of_equivalent_polynomial(self):
        record = {
            "dataset": "MATH",
            "final_answer": "2*x^2 + -2*x + 3",
            "ground_truth_solution": "Therefore the answer is \\boxed{2x^2 - 2x + 3}",
        }

        evaluation = evaluate_record(record)

        self.assertEqual(evaluation["evaluation_status"], "evaluated")
        self.assertTrue(evaluation["is_correct"])

    def test_math_evaluation_extracts_rhs_from_program_output(self):
        record = {
            "dataset": "MATH",
            "final_answer": "g(f(-1)): 2",
            "ground_truth_solution": "Therefore the answer is \\boxed{2}",
            "gold_answer": "2",
        }

        evaluation = evaluate_record(record)

        self.assertEqual(evaluation["evaluation_status"], "evaluated")
        self.assertTrue(evaluation["is_correct"])
        self.assertEqual(evaluation["predicted_answer"], "2")

    def test_math_evaluation_matches_number_words(self):
        record = {
            "dataset": "MATH",
            "final_answer": "two",
            "ground_truth_solution": "Therefore the answer is \\boxed{2}",
            "gold_answer": "2",
        }

        evaluation = evaluate_record(record)

        self.assertEqual(evaluation["evaluation_status"], "evaluated")
        self.assertTrue(evaluation["is_correct"])

    def test_math_evaluation_matches_latex_and_plain_fraction_forms(self):
        record = {
            "dataset": "MATH",
            "final_answer": "2/3",
            "ground_truth_solution": "Therefore the answer is \\boxed{\\frac{2}{3}}",
            "gold_answer": "\\frac{2}{3}",
        }

        evaluation = evaluate_record(record)

        self.assertEqual(evaluation["evaluation_status"], "evaluated")
        self.assertTrue(evaluation["is_correct"])

    def test_math_evaluation_accepts_reordered_commutative_factors(self):
        record = {
            "dataset": "MATH",
            "final_answer": "7(x-3)(x+3)",
            "ground_truth_solution": "Therefore the answer is \\boxed{7(x+3)(x-3)}",
        }

        evaluation = evaluate_record(record)

        self.assertEqual(evaluation["evaluation_status"], "evaluated")
        self.assertTrue(evaluation["is_correct"])
        self.assertEqual(evaluation["evaluation_method"], "math-symbolic")

    def test_unknown_dataset_uses_generic_semantic_matching(self):
        record = {
            "dataset": "Unknown",
            "final_answer": "The greatest possible value of a is: two",
            "ground_truth_solution": "2",
            "gold_answer": "2",
        }

        evaluation = evaluate_record(record)

        self.assertEqual(evaluation["evaluation_status"], "evaluated")
        self.assertTrue(evaluation["is_correct"])

    def test_unknown_dataset_matches_precise_numeric_output_to_rounded_display_value(self):
        record = {
            "dataset": "Unknown",
            "final_answer": "Circumference: 5.34070751110265",
            "ground_truth_solution": "5.34",
            "gold_answer": "5.34",
        }

        evaluation = evaluate_record(record)

        self.assertEqual(evaluation["evaluation_status"], "evaluated")
        self.assertTrue(evaluation["is_correct"])
        self.assertEqual(evaluation["evaluation_method"], "math-numeric")

    def test_summarize_accuracy_counts_evaluated_and_correct(self):
        summary = summarize_accuracy(
            [
                {"evaluation_status": "evaluated", "is_correct": True},
                {"evaluation_status": "evaluated", "is_correct": False},
                {"evaluation_status": "not-evaluated", "is_correct": None},
            ]
        )

        self.assertEqual(summary["evaluated"], 2)
        self.assertEqual(summary["correct"], 1)
        self.assertAlmostEqual(summary["accuracy"], 0.5)

    def test_aggregate_benchmark_resets_to_current_run_snapshot(self):
        row = {
            "timestamp": "2026-04-02T12:00:00",
            "label": "run_two",
            "model": "pg_walpha_sg",
            "dataset": "GSM",
            "evaluated": 6,
            "correct": 4,
            "accuracy": 4 / 6,
            "total_records": 6,
            "split": "test",
        }

        with patch("builtins.open", mock_open()):
            path, aggregate = update_aggregate_benchmark(TESTS_DIR, row)

        self.assertTrue(path.endswith("benchmark_summary.csv"))
        self.assertEqual(aggregate["run_count"], 1)
        self.assertEqual(aggregate["total_evaluated"], 6)
        self.assertEqual(aggregate["total_correct"], 4)
        self.assertAlmostEqual(aggregate["accuracy"], 4 / 6)


if __name__ == "__main__":
    unittest.main()

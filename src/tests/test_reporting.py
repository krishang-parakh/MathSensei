import os
import sys
import tempfile
import unittest


TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(TESTS_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from presentation.reporting import (
    _clean_code_display,
    classify_record,
    _detail_block,
    _dataset_filter_config,
    _dataset_distribution_values,
    _distribution_values,
    _incorrect_evaluation_panel,
    _level_sort_key,
    _looks_like_structured_literal_text,
    _record_sort_key,
    _render_answer_like_html,
    _render_question_text_html,
    _render_text_html,
    build_summary,
    generate_report,
    infer_dataset,
    infer_final_answer,
    load_records,
    normalize_record,
)


class TestReporting(unittest.TestCase):
    def test_normalize_record_keeps_module_order_and_kr_steps(self):
        record = {
            "pid": 0,
            "dataset": "GSM",
            "modules": ["knowledge_retrieval", "wolfram_alpha_search", "solution_generator"],
            "example": {
                "dataset": "GSM",
                "question": "A book costs $5 and a pen costs $2. What is the total?",
                "problem": "A book costs $5 and a pen costs $2. What is the total?",
                "solution": "#### 7",
            },
            "knowledge_retrieval:input": "Question: ...\nKnowledge Retrieval:",
            "knowledge_retrieval:output": "Add the two prices.",
            "query_generator:input": "Question: ...\nThought:",
            "query_generator:output": "Final Query: 5 + 2",
            "wolfram_alpha_search:input": "5 + 2",
            "wolfram_alpha_search:output": "7",
            "solution_generator:input": "Question: ...\nSolution:",
            "solution_generator:output": "We add 5 and 2.\n#### 7",
            "solution": "We add 5 and 2.\n#### 7",
            "answer": "#### 7",
        }

        normalized = normalize_record(record)

        self.assertEqual(
            normalized["modules_used"],
            ["knowledge_retrieval", "wolfram_alpha_search", "solution_generator"],
        )
        self.assertEqual(
            [step["title"] for step in normalized["module_steps"]],
            ["Knowledge Representation", "Wolfram Alpha Search", "Solution Generator"],
        )
        self.assertEqual(len(normalized["module_steps"][0]["items"]), 1)
        self.assertEqual(normalized["module_steps"][0]["items"][0]["value"], "Add the two prices.")
        self.assertEqual(normalized["final_answer"], "7")

    def test_normalize_record_preserves_multi_step_wolfram_trace(self):
        record = {
            "pid": 0,
            "dataset": "MMLU",
            "modules": ["wolfram_alpha_search", "solution_generator"],
            "example": {
                "dataset": "MMLU",
                "problem": "If 725x + 727y = 1500 and 729x + 731y = 1508, what is x-y?",
                "Answer": "D",
                "Option A": "725",
                "Option B": "-2",
                "Option C": "2",
                "Option D": "-48",
            },
            "wolfram_alpha_search:input": "Step 1: Solve[{725x + 727y == 1500, 729x + 731y == 1508}, {x, y}]\nStep 2: -23 - 25",
            "wolfram_alpha_search:output": "-48",
            "wolfram_alpha_search:trace": [
                {
                    "step": 1,
                    "thought": "Solve for x and y first.",
                    "query": "Solve[{725x + 727y == 1500, 729x + 731y == 1508}, {x, y}]",
                    "output": "x = -23 and y = 25",
                },
                {
                    "step": 2,
                    "thought": "Now compute x-y.",
                    "query": "-23 - 25",
                    "output": "-48",
                    "completion_thought": "The requested quantity is now satisfied.",
                },
            ],
            "solution_generator:output": "Final Answer: D",
            "solution": "Final Answer: D",
            "answer": "D",
        }

        normalized = normalize_record(record)
        wolfram_step = next(step for step in normalized["module_steps"] if step["module"] == "wolfram_alpha_search")
        labels = [item["label"] for item in wolfram_step["items"]]

        self.assertIn("Wolfram Step 1 Query", labels)
        self.assertIn("Wolfram Step 1 Output", labels)
        self.assertIn("Wolfram Step 2 Query", labels)
        self.assertIn("Wolfram Step 2 Output", labels)
        self.assertIn("Wolfram Step 2 Completion Check", labels)

    def test_infer_final_answer_accepts_final_answer_label(self):
        record = {
            "final_generated_solution": "We simplify the expression.\nFinal Answer: \\frac{1}{2}",
            "options": None,
        }

        self.assertEqual(infer_final_answer(record), "\\frac{1}{2}")

    def test_infer_final_answer_extracts_rhs_from_program_output(self):
        record = {
            "program_output": "g(f(-1)): 2",
            "options": None,
        }

        self.assertEqual(infer_final_answer(record), "2")

    def test_infer_final_answer_prefers_explicit_final_answer_tag_for_mcq(self):
        record = {
            "dataset": "AQUA",
            "options": [
                {"key": "A", "label": "725"},
                {"key": "B", "label": "-2"},
                {"key": "C", "label": "2"},
                {"key": "D", "label": "-48"},
            ],
            "final_generated_solution": (
                "Step 4: Compute x - y.\n\n"
                "x - y = -23 - 25 = -48\n\n"
                "Alternatively, since x + y = 2, and x - y is what we need, the direct calculation confirms the value.\n\n"
                "Final Answer: D"
            ),
        }

        self.assertEqual(infer_final_answer(record), "D. -48")

    def test_infer_final_answer_uses_explicit_option_conclusion_instead_of_rejected_checklist_line(self):
        record = {
            "dataset": "MMLU",
            "options": [
                {"key": "A", "label": "5"},
                {"key": "B", "label": "6"},
                {"key": "C", "label": "7"},
                {"key": "D", "label": "8"},
            ],
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
        }

        self.assertEqual(infer_final_answer(record), "A. 5")

    def test_infer_final_answer_ignores_trailing_option_block_after_explicit_mcq_conclusion(self):
        record = {
            "dataset": "AQUA",
            "options": [
                {"key": "A", "label": "30 feet"},
                {"key": "B", "label": "20 feet"},
                {"key": "C", "label": "10 feet"},
                {"key": "D", "label": "50 feet"},
                {"key": "E", "label": "60 feet"},
            ],
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
        }

        self.assertEqual(infer_final_answer(record), "A. 30 feet")

    def test_infer_final_answer_prefers_resolved_option_value_over_conflicting_explicit_display(self):
        record = {
            "dataset": "AQUA",
            "problem": "At a certain factory, 10 percent of the staplers produced on Monday were defective and 2 percent of the non-defective staplers were rejected by mistake. If 72 of the non-defective staplers were rejected, what was the number of staplers produced that day?",
            "final_answer": "E. 5,000",
            "final_generated_solution": (
                "0.018N = 72\n\n"
                "N = 72 / 0.018 = 4000\n\n"
                "Thus, the total number of staplers produced that day is 4000.\n\n"
                "The answer is E."
            ),
            "program_output": "Number of staplers produced that day: 4000.00000000000\n",
            "wolfram_output": "4,000",
            "options": [
                {"key": "A", "label": "4,000"},
                {"key": "B", "label": "4,200"},
                {"key": "C", "label": "4,500"},
                {"key": "D", "label": "4,800"},
                {"key": "E", "label": "5,000"},
            ],
        }

        self.assertEqual(infer_final_answer(record), "A. 4,000")

    def test_normalize_record_keeps_mcq_answer_correct_when_option_block_pushes_correct_line_out_of_tail(self):
        record = {
            "dataset": "AQUA",
            "modules": ["solution_generator"],
            "example": {
                "dataset": "AQUA",
                "question": (
                    "Julie's yard is rectangular. One side of the yard is 100 feet wide. "
                    "The total area of the yard is 3,000 square feet. What is the length of the other side of the yard?"
                ),
                "problem": (
                    "Julie's yard is rectangular. One side of the yard is 100 feet wide. "
                    "The total area of the yard is 3,000 square feet. What is the length of the other side of the yard?"
                ),
                "options": ["A) 30 feet", "B) 20 feet", "C) 10 feet", "D) 50 feet", "E) 60 feet"],
                "correct": "A",
            },
            "solution_generator:output": (
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
            "solution": (
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
        }

        normalized = normalize_record(record)

        self.assertEqual(normalized["final_answer"], "A. 30 feet")
        self.assertEqual(normalized["final_answer"], normalized["predicted_answer_display"])
        self.assertEqual(normalized["predicted_answer"], "30 feet")
        self.assertEqual(normalized["gold_answer"], "30 feet")
        self.assertEqual(normalized["gold_answer_display"], "A. 30 feet")
        self.assertTrue(normalized["is_correct"])
        method_rows = {row["label"]: row for row in normalized["method_evaluations"]}
        self.assertEqual(method_rows["Solution Generator"]["answer"], "A. 30 feet")

    def test_normalize_record_does_not_treat_math_coefficients_as_mcq_answer(self):
        record = {
            "pid": 11,
            "dataset": "MATH",
            "problem": "When the polynomial f(x) is divided by x - 3, the remainder is 15. When f(x) is divided by (x - 1)^2, the remainder is 2x + 1. Find the remainder when f(x) is divided by (x - 3)(x - 1)^2.",
            "ground_truth_solution": "The remainder is \\boxed{2x^2 - 2x + 3}.",
            "program_error": "SyntaxError: invalid syntax (<string>, line 28)",
            "wolfram_output": "\\(2x^2 - 2x + 3\\)",
            "final_generated_solution": (
                "A + B + C = 3\n"
                "2A + B = 2\n"
                "9A + 3B + C = 15\n"
                "Final remainder polynomial: \\boxed{2x^2 - 2x + 3}"
            ),
            "modules_used": ["program_executor", "wolfram_alpha_search", "solution_generator"],
        }

        normalized = normalize_record(record)

        self.assertIsNone(normalized["correct_option"])
        self.assertEqual(normalized["gold_answer"], "2x^2 - 2x + 3")
        self.assertEqual(normalized["final_answer"], "2x^2 - 2x + 3")

        method_evaluations = {item["label"]: item for item in normalized["method_evaluations"]}
        self.assertEqual(method_evaluations["Python"]["status"], "error")
        self.assertEqual(method_evaluations["Wolfram"]["status"], "correct")
        self.assertEqual(normalized["status"], "complete")

    def test_normalize_record_evaluates_gsm_sentence_answers_and_bold_final_answer(self):
        record = {
            "dataset": "GSM",
            "problem": "Although Soledad works in a windowless office, she loves the outdoors. She will be on vacation for the entire month of June and cannot wait to hike 9300 miles within that month. She is thinking of walking twice a day, covering 125 miles each time. How many more miles per day must Soledad hike to complete her journey on time?",
            "ground_truth_solution": "June has 30 days.\n#### 60",
            "gold_answer": "60",
            "program_output": "51.666666666666664\n",
            "wolfram_output": "She must hike 60 miles more per day.",
            "final_generated_solution": "**Final answer: 0**",
            "modules_used": ["program_executor", "wolfram_alpha_search", "solution_generator"],
        }

        normalized = normalize_record(record)
        method_evaluations = {item["label"]: item for item in normalized["method_evaluations"]}

        self.assertEqual(normalized["final_answer"], "0")
        self.assertEqual(normalized["overall_evaluation_label"], "Incorrect")
        self.assertEqual(method_evaluations["Solution Generator"]["status"], "incorrect")
        self.assertEqual(method_evaluations["Python"]["status"], "incorrect")
        self.assertEqual(method_evaluations["Wolfram"]["status"], "correct")

    def test_normalize_record_prefers_mcq_value_over_conflicting_bare_option_letter(self):
        record = {
            "dataset": "AQUA",
            "problem": "At a certain factory, 10 percent of the staplers produced on Monday were defective and 2 percent of the non-defective staplers were rejected by mistake. If 72 of the non-defective staplers were rejected, what was the number of staplers produced that day?",
            "ground_truth_solution": "Final Answer:\nA",
            "gold_answer": "A",
            "options": [
                {"key": "A", "label": "4,000"},
                {"key": "B", "label": "4,200"},
                {"key": "C", "label": "4,500"},
                {"key": "D", "label": "4,800"},
                {"key": "E", "label": "5,000"},
            ],
            "final_answer": "E. 5,000",
            "final_generated_solution": (
                "0.018N = 72\n\n"
                "N = 72 / 0.018 = 4000\n\n"
                "Thus, the total number of staplers produced that day is 4000.\n\n"
                "The answer is E."
            ),
            "program_output": "Number of staplers produced that day: 4000.00000000000\n",
            "wolfram_output": "4,000",
            "modules_used": ["program_executor", "wolfram_alpha_search", "solution_generator"],
        }

        normalized = normalize_record(record)
        method_evaluations = {item["label"]: item for item in normalized["method_evaluations"]}

        self.assertEqual(normalized["final_answer"], "A. 4,000")
        self.assertEqual(normalized["overall_evaluation_label"], "Correct")
        self.assertEqual(method_evaluations["Solution Generator"]["status"], "correct")

    def test_program_steps_preserve_concise_python_comments(self):
        record = normalize_record(
            {
                "dataset": "GSM",
                "problem": "A book costs 5 and a pen costs 2. What is the total?",
                "ground_truth_solution": "#### 7",
                "gold_answer": "7",
                "generated_program": "from sympy import *\n# Add the item costs\nbook = 5\npen = 2\nprint(book + pen)",
                "program_output": "7",
                "final_answer": "7",
                "modules_used": ["program_generator", "program_executor", "solution_generator"],
            }
        )

        program_generator_step = next(step for step in record["module_steps"] if step["module"] == "program_generator")
        program_executor_step = next(step for step in record["module_steps"] if step["module"] == "program_executor")

        generated_program_item = next(item for item in program_generator_step["items"] if item["label"] == "Generated Program")
        executed_program_item = next(item for item in program_executor_step["items"] if item["label"] == "Executed Program")

        self.assertIn("# Add the item costs", generated_program_item["value"])
        self.assertFalse(generated_program_item["strip_comments"])
        self.assertIn("# Add the item costs", executed_program_item["value"])
        self.assertFalse(executed_program_item["strip_comments"])

    def test_render_text_keeps_currency_and_converts_actual_inline_math(self):
        rendered = _render_text_html(
            "The price goes from $5 to $7 while $x^2 + 1$ stays symbolic.",
            preserve_breaks=False,
        )

        self.assertIn("$5", rendered)
        self.assertIn("$7", rendered)
        self.assertIn(r"\(x^2 + 1\)", rendered)
        self.assertNotIn("$x^2 + 1$", rendered)

    def test_render_text_strips_markdown_heading_noise(self):
        rendered = _render_text_html(
            "### Knowledge Retrieval for Question 3\n**Reasoning**\nThe total is $5.",
            preserve_breaks=True,
        )

        self.assertNotIn("###", rendered)
        self.assertIn("Knowledge Retrieval for Question 3", rendered)
        self.assertNotIn("**", rendered)
        self.assertIn("$5", rendered)

    def test_render_text_converts_symbolic_math_tokens(self):
        rendered = _render_text_html(
            r"Each of the symbols $\star$ and $*$ is allowed.",
            preserve_breaks=False,
        )

        self.assertIn(r"\(\star\)", rendered)
        self.assertIn(r"\(*\)", rendered)
        self.assertNotIn(r"$\star$", rendered)
        self.assertNotIn("$*$", rendered)

    def test_render_text_converts_single_letter_and_segment_labels(self):
        rendered = _render_text_html(
            r"Let $A$ and $B$ lie on $y^2 = 4x$. The circle with diameter $AB$ has radius $r$.",
            preserve_breaks=False,
        )

        self.assertIn(r"\(A\)", rendered)
        self.assertIn(r"\(B\)", rendered)
        self.assertIn(r"\(AB\)", rendered)
        self.assertIn(r"\(r\)", rendered)
        self.assertIn(r"\(y^2 = 4x\)", rendered)
        self.assertNotIn("$A$", rendered)
        self.assertNotIn("$AB$", rendered)

    def test_render_text_converts_function_notation_inline_math(self):
        rendered = _render_text_html(
            r"When $f(x)$ is divided by $(x-1)^2$, evaluate $g(f(x))$.",
            preserve_breaks=False,
        )

        self.assertIn(r"\(f(x)\)", rendered)
        self.assertIn(r"\((x-1)^2\)", rendered)
        self.assertIn(r"\(g(f(x))\)", rendered)
        self.assertNotIn("$f(x)$", rendered)
        self.assertNotIn("$g(f(x))$", rendered)

    def test_render_text_converts_polygon_labels(self):
        rendered = _render_text_html(
            r"A regular octagon $ABCDEFGH$ has area one. Find the area of rectangle $ABEF$.",
            preserve_breaks=False,
        )

        self.assertIn(r"\(ABCDEFGH\)", rendered)
        self.assertIn(r"\(ABEF\)", rendered)
        self.assertNotIn("$ABCDEFGH$", rendered)
        self.assertNotIn("$ABEF$", rendered)

    def test_render_text_preserves_multiple_currency_amounts(self):
        rendered = _render_text_html(
            "The prices are $5, $7, and $12.50 while $x+1$ stays symbolic.",
            preserve_breaks=False,
        )

        self.assertIn("$5", rendered)
        self.assertIn("$7", rendered)
        self.assertIn("$12.50", rendered)
        self.assertIn(r"\(x+1\)", rendered)
        self.assertNotIn(r"\(5, 7, and 12.50 while\)", rendered)

    def test_render_text_treats_numeric_prefixed_equations_as_math_not_currency(self):
        rendered = _render_text_html(
            r"If $725x + 727y = 1500$ and $729x + 731y = 1508$, what is $x-y$?",
            preserve_breaks=False,
        )

        self.assertIn(r"\(725x + 727y = 1500\)", rendered)
        self.assertIn(r"\(729x + 731y = 1508\)", rendered)
        self.assertIn(r"\(x-y\)", rendered)
        self.assertNotIn("$725x", rendered)
        self.assertNotIn("$729x", rendered)

    def test_render_question_text_formats_translation_key_without_auto_math_wrapping(self):
        question = (
            " Select the best English interpretation of the given proposition, using the following translation key: "
            "Ax: x is an apartment Hx: x is a house Lx: x is large Bxy: x is bigger than y "
            "(\u2200x){Ax \u2283 (\u2200y)[(Hy \u2022 Ly) \u2283 \u223cBxy]}"
        )

        rendered = _render_question_text_html(question, preserve_breaks=True)

        self.assertIn("translation key:<br>Ax: x is an apartment<br>Hx: x is a house", rendered)
        self.assertIn("<br>Bxy: x is bigger than y<br>(\u2200x){Ax \u2283 (\u2200y)[(Hy \u2022 Ly) \u2283 \u223cBxy]}", rendered)
        self.assertNotIn(r"\(", rendered)

    def test_render_question_text_keeps_explicit_inline_math_delimiters(self):
        rendered = _render_question_text_html(
            "If $x^2 + 1$ equals 10, what is x?",
            preserve_breaks=False,
        )

        self.assertIn(r"\(x^2 + 1\)", rendered)
        self.assertNotIn("$x^2 + 1$", rendered)

    def test_render_text_replaces_asy_block_with_svg_diagram(self):
        rendered = _render_text_html(
            """Let the rectangle be shown below.

[asy]
size(100);
draw((0,0)--(8,0)--(8,10)--(0,10)--cycle);
dot((8,10)); dot((0,6)); dot((3,10));
label("$8''$", (0,0)--(8,0), S);
draw((8,0)--(3,10), dashed);
draw((0,6)--(3,10)); draw((0,6)--(8,0));
[/asy]""",
            preserve_breaks=True,
        )

        self.assertIn("asy-diagram__svg", rendered)
        self.assertIn("asy-diagram__expand", rendered)
        self.assertIn("<svg", rendered)
        self.assertIn("8&#x27;&#x27;", rendered)
        self.assertNotIn("[asy]", rendered)
        self.assertNotIn("Diagram Source", rendered)
        self.assertNotIn("draw((0,0)", rendered)
        self.assertNotIn('font-size="13"', rendered)
        self.assertNotIn('r="3.400"', rendered)

    def test_render_text_renders_asy_graph_blocks_without_fallback(self):
        rendered = _render_text_html(
            """Graph the parabola.

[asy]
Label f;
f.p=fontsize(4);
xaxis(-1,3,Ticks(f, 1.0));
yaxis(-1,3,Ticks(f, 1.0));
real g(real x)
{
return (x-1)^2;
}
draw(graph(g,-1,3), red);
[/asy]""",
            preserve_breaks=True,
        )

        self.assertIn("asy-diagram__svg", rendered)
        self.assertIn("<svg", rendered)
        self.assertIn("asy-arrow-end", rendered)
        self.assertNotIn("Diagram Source", rendered)

    def test_clean_code_display_removes_fences_and_comments(self):
        cleaned = _clean_code_display(
            """```python
# Python generator
from sympy import *  # import everything
value = 3
/* remove this block */
print(value)  # final
```"""
        )

        self.assertNotIn("```", cleaned)
        self.assertNotIn("# Python generator", cleaned)
        self.assertNotIn("# import everything", cleaned)
        self.assertNotIn("/*", cleaned)
        self.assertEqual(cleaned, "from sympy import *\nvalue = 3\nprint(value)")

    def test_structured_literal_text_is_detected_and_rendered_as_code(self):
        json_text = '{"equation": "$$\\\\sqrt{169} = 13$$", "final_answer": 13}'
        python_text = "{'equation': '$x^2 + 1$', 'note': 'literal text'}"

        self.assertTrue(_looks_like_structured_literal_text(json_text))
        self.assertTrue(_looks_like_structured_literal_text(python_text))

        rendered = _detail_block("JSON", json_text, code=False)

        self.assertIn('<pre class="code-block">', rendered)
        self.assertIn("$$\\\\sqrt{169} = 13$$", rendered)
        self.assertNotIn("\\[", rendered)

    def test_render_text_keeps_structured_literal_delimiters_literal(self):
        structured_text = '{"problem": "Each symbol $$\\\\star$$ or $*$ stays literal here."}'

        rendered = _render_text_html(structured_text, preserve_breaks=False)

        self.assertIn("$$\\\\star$$", rendered)
        self.assertIn("$*$", rendered)
        self.assertNotIn("\\(", rendered)
        self.assertNotIn("\\[", rendered)

    def test_render_answer_like_html_canonicalizes_python_sign_sequences(self):
        rendered = _render_answer_like_html("2 * x^2 + -2 * x + 3")

        self.assertNotIn("+-", rendered)
        self.assertNotIn("+ -", rendered)
        self.assertIn(r"\(", rendered)
        self.assertIn("x^{2}", rendered)
        self.assertIn("- 2 x", rendered)

    def test_render_text_wraps_raw_latex_set_notation(self):
        rendered = _render_text_html(
            r"Final Answer: (-\infty,-2]\cup[2,\infty)",
            preserve_breaks=False,
        )

        self.assertIn(r"Final Answer: \(", rendered)
        self.assertIn(r"\infty", rendered)
        self.assertIn(r"\cup", rendered)

    def test_normalize_record_extracts_options_cleanly(self):
        record = {
            "pid": 1,
            "dataset": "MMLU",
            "modules": ["solution_generator"],
            "example": {
                "dataset": "MMLU",
                "Question": "Which choice is correct?",
                "Option A": "Alpha",
                "Option B": "Beta",
                "Option C": "Gamma",
                "Option D": "Delta",
                "problem": "\nWhich choice is correct?\nOption A:Alpha\nOption B:Beta\nOption C:Gamma\nOption D:Delta",
                "Answer": "B",
            },
            "solution_generator:output": "The correct option is #### B",
            "solution": "The correct option is #### B",
            "answer": "#### B",
        }

        normalized = normalize_record(record)

        self.assertEqual(normalized["problem"], "Which choice is correct?")
        self.assertEqual(len(normalized["options"]), 4)
        self.assertEqual(normalized["options"][1]["key"], "B")
        self.assertEqual(normalized["options"][1]["label"], "Beta")
        self.assertEqual(normalized["correct_option"], "B")

    def test_infer_dataset_returns_mixed_for_multi_dataset_records(self):
        records = [
            {"dataset": "GSM", "problem_type": "Unknown", "level": "Unknown", "options": []},
            {"dataset": "MMLU", "problem_type": "biology", "level": "Unknown", "options": [{"key": "A", "label": "one"}]},
        ]

        self.assertEqual(infer_dataset(records=records), "MIXED")

    def test_dataset_filter_config_includes_dataset_filter_for_mixed_runs(self):
        records = [
            {"dataset": "GSM", "problem_type": "Arithmetic", "level": "Easy"},
            {"dataset": "MMLU", "problem_type": "Biology", "level": "Unknown"},
            {"dataset": "AQUA", "problem_type": "Algebra", "level": "Unknown"},
        ]

        config = _dataset_filter_config("MIXED", records)

        dataset_filter = next(item for item in config if item["id"] == "datasetFilter")
        self.assertEqual(dataset_filter["label"], "Dataset")
        self.assertEqual(dataset_filter["attr"], "data-dataset")
        self.assertEqual(
            dataset_filter["options"],
            [
                {"value": "AQUA", "label": "AQUA"},
                {"value": "GSM", "label": "GSM"},
                {"value": "MMLU", "label": "MMLU"},
            ],
        )

    def test_dataset_filter_config_sorts_levels_numerically(self):
        records = [
            {"dataset": "MATH", "problem_type": "Algebra", "level": "Level 5"},
            {"dataset": "MATH", "problem_type": "Algebra", "level": "Level 2"},
            {"dataset": "MATH", "problem_type": "Algebra", "level": "Level 1"},
        ]

        config = _dataset_filter_config("MATH", records)

        level_filter = next(item for item in config if item["id"] == "levelFilter")
        self.assertEqual(
            level_filter["options"],
            [
                {"value": "Level 1", "label": "Level 1"},
                {"value": "Level 2", "label": "Level 2"},
                {"value": "Level 5", "label": "Level 5"},
            ],
        )

    def test_record_sort_key_uses_dataset_type_then_level(self):
        first = {"dataset": "AQUA", "problem_type": "Algebra", "level": "Level 2", "pid": 3}
        second = {"dataset": "GSM", "problem_type": "Arithmetic", "level": "Level 1", "pid": 0}
        third = {"dataset": "AQUA", "problem_type": "Algebra", "level": "Level 5", "pid": 1}

        ordered = sorted([second, third, first], key=_record_sort_key)

        self.assertEqual(ordered, [first, third, second])

    def test_distribution_values_can_sort_levels_numerically(self):
        records = [
            {"level": "Level 5"},
            {"level": "Level 1"},
            {"level": "Level 2"},
        ]

        values = _distribution_values(records, lambda record: record.get("level"), sort_key=_level_sort_key)
        self.assertEqual(list(values.keys()), ["Level 1", "Level 2", "Level 5"])

    def test_build_summary_tracks_incorrect_evaluation_count(self):
        records = [
            {"status": "complete", "evaluation_status": "evaluated", "is_correct": True, "problem_type": "Algebra", "level": "Level 1", "dataset": "MATH"},
            {"status": "incorrect-evaluation", "evaluation_status": "evaluated", "is_correct": False, "problem_type": "Algebra", "level": "Level 2", "dataset": "GSM"},
            {"status": "needs-review", "evaluation_status": "not-evaluated", "is_correct": None, "problem_type": "Geometry", "level": "Level 3", "dataset": "MATH"},
        ]

        summary = build_summary(records)

        self.assertEqual(summary["incorrect_evaluation"], 1)
        self.assertEqual(summary["needs_review"], 1)
        self.assertNotIn("incomplete_evaluation", summary)

    def test_overall_evaluation_uses_final_answer_not_all_method_outputs(self):
        record = normalize_record(
            {
                "dataset": "GSM",
                "problem": "A total should be 14.",
                "ground_truth_solution": "#### 14",
                "gold_answer": "14",
                "program_output": "12",
                "wolfram_output": "13",
                "solution_generator_output": "We reason correctly.\n#### 14",
                "final_generated_solution": "We reason correctly.\n#### 14",
                "final_answer": "#### 14",
                "modules_used": ["program_executor", "wolfram_alpha_search", "solution_generator"],
            }
        )

        method_rows = {row["label"]: row for row in record["method_evaluations"]}

        self.assertEqual(record["overall_evaluation_label"], "Correct")
        self.assertEqual(record["status"], "complete")
        self.assertEqual(method_rows["Python"]["status"], "incorrect")
        self.assertEqual(method_rows["Wolfram"]["status"], "incorrect")
        self.assertEqual(method_rows["Solution Generator"]["status"], "correct")

    def test_classify_record_no_longer_returns_partial(self):
        record = {
            "evaluation_status": "not-evaluated",
            "is_correct": None,
            "program_error": "",
            "wolfram_output": "",
            "final_generated_solution": "",
            "program_output": "",
            "final_answer": None,
        }

        self.assertEqual(classify_record(record), "needs-review")

    def test_dataset_distribution_values_include_zero_count_datasets(self):
        records = [
            {"dataset": "GSM"},
            {"dataset": "GSM"},
            {"dataset": "MATH"},
        ]

        values = _dataset_distribution_values(records)

        self.assertEqual(
            values,
            {
                "AQUA": 0,
                "GSM": 2,
                "MATH": 1,
                "MMLU": 0,
            },
        )

    def test_incorrect_evaluation_panel_only_shows_one_line_reason(self):
        record = normalize_record(
            {
                "dataset": "GSM",
                "problem": "A total should be 14.",
                "ground_truth_solution": "#### 14",
                "gold_answer": "14",
                "program_output": "14",
                "solution_generator_output": "We reason incorrectly.\n#### 12",
                "final_generated_solution": "We reason incorrectly.\n#### 12",
                "final_answer": "#### 12",
                "modules_used": ["program_executor", "solution_generator"],
            }
        )

        panel = _incorrect_evaluation_panel(record)

        self.assertIn("Incorrect Answer", panel)
        self.assertIn("Solution Generator", panel)
        self.assertNotIn("Final Answer</span>", panel)
        self.assertNotIn("Gold Answer</span>", panel)
        self.assertNotIn("Final Method", panel)
        self.assertNotIn("Gold Method", panel)
        self.assertNotIn("Correct</span>", panel)

    def test_program_error_is_flagged_per_method_but_final_correct_record_is_complete(self):
        record = normalize_record(
            {
                "dataset": "MATH",
                "problem": "Compute sin 120 degrees.",
                "ground_truth_solution": r"\frac{\sqrt{3}}{2}",
                "gold_answer": r"\frac{\sqrt{3}}{2}",
                "program_error": "name 'radians' is not defined",
                "wolfram_output": r"\frac{\sqrt{3}}{2}",
                "solution_generator_output": r"The answer is \frac{\sqrt{3}}{2}.",
                "final_generated_solution": r"The answer is \frac{\sqrt{3}}{2}.",
                "final_answer": r"\frac{\sqrt{3}}{2}",
            }
        )

        method_rows = {row["label"]: row for row in record["method_evaluations"]}

        self.assertEqual(method_rows["Python"]["status"], "error")
        self.assertEqual(record["status"], "complete")
        self.assertEqual(classify_record(record), "complete")

    def test_load_records_skips_malformed_jsonl_rows(self):
        valid_row = '{"dataset": "GSM", "problem": "Q", "ground_truth_solution": "#### 1", "final_answer": "1"}\n'
        invalid_row = '{"dataset": "GSM", "problem": "broken"\n'

        with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8") as handle:
            handle.write(valid_row)
            handle.write(invalid_row)
            temp_path = handle.name

        try:
            records, skipped_rows = load_records(temp_path)
        finally:
            os.unlink(temp_path)

        self.assertEqual(len(records), 1)
        self.assertEqual(skipped_rows, 1)

    def test_generate_report_writes_html_file(self):
        valid_row = '{"dataset": "GSM", "problem": "Q", "ground_truth_solution": "#### 1", "gold_answer": "1", "final_answer": "1"}\n'

        with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8", suffix=".jsonl") as input_handle:
            input_handle.write(valid_row)
            input_path = input_handle.name

        with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8", suffix=".html") as output_handle:
            output_path = output_handle.name

        try:
            generated_path, total = generate_report(input_path, output_path=output_path, title="MathSensei")
            with open(generated_path, "r", encoding="utf-8") as handle:
                html = handle.read()
        finally:
            os.unlink(input_path)
            os.unlink(output_path)

        self.assertEqual(total, 1)
        self.assertEqual(generated_path, output_path)
        self.assertIn("<!DOCTYPE html>", html)
        self.assertIn("MathSensei", html)
        self.assertNotIn('<div class="hero-heading__subtitle">', html)
        self.assertIn('id="resultsSummary"', html)
        self.assertIn('id="resultsVisibleCount">1<', html)
        self.assertIn('id="resultsTotalCount">1<', html)
        self.assertIn("All problems shown", html)

    def test_generate_report_exposes_method_filters_and_hides_incomplete_evaluation_status(self):
        valid_row = (
            '{"dataset": "GSM", "problem": "A total should be 14.", '
            '"ground_truth_solution": "#### 14", "gold_answer": "14", '
            '"program_output": "12", "solution_generator_output": "We reason correctly.\\n#### 14", '
            '"final_generated_solution": "We reason correctly.\\n#### 14", "final_answer": "#### 14", '
            '"modules_used": ["program_executor", "solution_generator"]}\n'
        )

        with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8", suffix=".jsonl") as input_handle:
            input_handle.write(valid_row)
            input_path = input_handle.name

        with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8", suffix=".html") as output_handle:
            output_path = output_handle.name

        try:
            generated_path, _ = generate_report(input_path, output_path=output_path, title="MathSensei")
            with open(generated_path, "r", encoding="utf-8") as handle:
                html = handle.read()
        finally:
            os.unlink(input_path)
            os.unlink(output_path)

        self.assertIn('data-attr="data-solution-generator-status"', html)
        self.assertIn('data-attr="data-python-status"', html)
        self.assertIn('value="flagged">Flagged<', html)
        self.assertIn('data-solution-generator-status="correct"', html)
        self.assertIn('data-python-status="incorrect"', html)
        self.assertNotIn('value="incomplete-evaluation"', html)
        self.assertNotIn(">Incomplete Evaluation<", html)

    def test_load_records_dedupes_repeated_question_rows_by_signature(self):
        stale_row = (
            '{"pid": 7, "dataset": "GSM", "question_signature": "abc123", '
            '"problem": "How many apples are left?", "ground_truth_solution": "#### 4", '
            '"gold_answer": "4", "final_answer": "9"}\n'
        )
        fresh_row = (
            '{"pid": 7, "dataset": "GSM", "question_signature": "abc123", '
            '"problem": "How many apples are left?", "ground_truth_solution": "#### 4", '
            '"gold_answer": "4", "final_answer": "4"}\n'
        )

        with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8", suffix=".jsonl") as handle:
            handle.write(stale_row)
            handle.write(fresh_row)
            temp_path = handle.name

        try:
            records, skipped_rows = load_records(temp_path)
        finally:
            os.unlink(temp_path)

        self.assertEqual(skipped_rows, 0)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["final_answer"], "4")
        self.assertEqual(records[0]["overall_evaluation_label"], "Correct")

    def test_generate_report_includes_compact_asy_styles(self):
        valid_row = '{"dataset": "GSM", "problem": "[asy]\\nsize(100);\\ndraw((0,0)--(4,0)--(4,3)--cycle);\\n[/asy]", "ground_truth_solution": "#### 1", "gold_answer": "1", "final_answer": "1"}\n'

        with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8", suffix=".jsonl") as input_handle:
            input_handle.write(valid_row)
            input_path = input_handle.name

        with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8", suffix=".html") as output_handle:
            output_path = output_handle.name

        try:
            generated_path, _ = generate_report(input_path, output_path=output_path, title="MathSensei")
            with open(generated_path, "r", encoding="utf-8") as handle:
                html = handle.read()
        finally:
            os.unlink(input_path)
            os.unlink(output_path)

        self.assertIn("width: min(100%, 360px);", html)
        self.assertIn("max-height: 220px;", html)
        self.assertIn('id="asyOverlay"', html)
        self.assertIn(".asy-overlay__content .asy-diagram__svg", html)
        self.assertIn("width: min(100%, 308px);", html)


if __name__ == "__main__":
    unittest.main()

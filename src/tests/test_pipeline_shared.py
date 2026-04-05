import os
import sys
import unittest


TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(TESTS_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from core.pipeline_shared import (
    build_initial_question_response,
    build_knowledge_response_context,
    build_wolfram_response_context,
    build_wolfram_next_step_prompt,
    detect_wolfram_loop_reason,
    extract_example_metadata,
    format_wolfram_trace,
    normalize_module_sequence,
    normalize_wolfram_output_signature,
    normalize_wolfram_query_signature,
    parse_wolfram_next_step_response,
    parse_wolfram_query_response,
    rebuild_cached_response,
)


class TestPipelineShared(unittest.TestCase):
    def test_parse_wolfram_query_response_extracts_thought_and_query(self):
        parsed = parse_wolfram_query_response(
            "Thought: Solve the system first.\nAnswer: Use Wolfram.\nFinal Query: Solve[{x+y==2,x-y==4},{x,y}]"
        )

        self.assertEqual(parsed["thought"], "Solve the system first.")
        self.assertEqual(parsed["query"], "Solve[{x+y==2,x-y==4},{x,y}]")

    def test_parse_wolfram_next_step_response_supports_continue(self):
        parsed = parse_wolfram_next_step_response(
            "Status: CONTINUE\nThought: We have x and y, but the question asks for x-y.\nNext Query: -23 - 25\nFinal Answer:\n"
        )

        self.assertEqual(parsed["status"], "CONTINUE")
        self.assertEqual(parsed["next_query"], "-23 - 25")
        self.assertIsNone(parsed["final_answer"])

    def test_parse_wolfram_next_step_response_supports_final(self):
        parsed = parse_wolfram_next_step_response(
            '{"status":"FINAL","thought":"The requested quantity is now computed.","final_answer":"-48"}'
        )

        self.assertEqual(parsed["status"], "FINAL")
        self.assertEqual(parsed["final_answer"], "-48")
        self.assertIsNone(parsed["next_query"])

    def test_build_wolfram_next_step_prompt_includes_trace(self):
        trace = [
            {"step": 1, "thought": "Solve for x and y.", "query": "Solve[{...},{x,y}]", "output": "x = -23 and y = 25"},
        ]

        prompt = build_wolfram_next_step_prompt("What is x - y?", trace)

        self.assertIn("What is x - y?", prompt)
        self.assertIn("Step 1:", prompt)
        self.assertIn("Query: Solve[{...},{x,y}]", prompt)
        self.assertIn("Output: x = -23 and y = 25", prompt)
        self.assertIn("compare absolute differences", prompt)

    def test_format_wolfram_trace_lists_step_fields(self):
        text = format_wolfram_trace(
            [
                {"step": 1, "thought": "Solve for x and y.", "query": "Solve[{...},{x,y}]", "output": "x = -23 and y = 25"},
                {"step": 2, "thought": "Compute x-y.", "query": "-23 - 25", "output": "-48"},
            ]
        )

        self.assertIn("Step 1:", text)
        self.assertIn("Step 2:", text)
        self.assertIn("Output: -48", text)

    def test_normalize_wolfram_query_signature_ignores_spacing_and_case(self):
        self.assertEqual(
            normalize_wolfram_query_signature(" Solve[ {x + y == 2, x - y == 4}, {x, y} ] "),
            normalize_wolfram_query_signature("solve[{x+y==2,x-y==4},{x,y}]"),
        )

    def test_normalize_wolfram_output_signature_ignores_spacing(self):
        self.assertEqual(
            normalize_wolfram_output_signature("x = -23 and y = 25"),
            normalize_wolfram_output_signature("x=-23 and y=25"),
        )

    def test_detect_wolfram_loop_reason_flags_repeated_query(self):
        trace = [
            {"step": 1, "query": "Solve[{x+y==2,x-y==4},{x,y}]", "output": "x = 3 and y = -1"},
        ]

        reason = detect_wolfram_loop_reason(trace, next_query=" solve[ {x+y==2, x-y==4}, {x,y} ] ")

        self.assertIn("repeated a previous query", reason)

    def test_detect_wolfram_loop_reason_flags_repeated_output(self):
        trace = [
            {"step": 1, "query": "Solve[{x+y==2,x-y==4},{x,y}]", "output": "x = 3 and y = -1"},
            {"step": 2, "query": "NSolve[{x+y==2,x-y==4},{x,y}]", "output": "x = 3 and y = -1"},
        ]

        reason = detect_wolfram_loop_reason(trace, next_query="x - y")

        self.assertIn("same output again", reason)

    def test_normalize_module_sequence_deduplicates_preserving_order(self):
        modules = normalize_module_sequence(
            ["knowledge_retrieval", "program_generator", "program_executor", "program_executor", "solution_generator", "solution_generator"]
        )

        self.assertEqual(
            modules,
            ["knowledge_retrieval", "program_generator", "program_executor", "solution_generator"],
        )

    def test_build_initial_question_response_adds_math_metadata(self):
        response = build_initial_question_response({"type": "Geometry", "level": "Level 3"}, "MATH")

        self.assertIn("Mathematics Problem Type:Geometry", response)
        self.assertIn("Level of Problem:Level 3", response)

    def test_extract_example_metadata_tolerates_missing_subject_type_and_level(self):
        metadata = extract_example_metadata({"problem": "I have 2 faces, no vertices, and I can roll."})

        self.assertEqual(metadata["topic"], "")
        self.assertEqual(metadata["level"], "")

    def test_extract_example_metadata_uses_available_fallback_fields(self):
        metadata = extract_example_metadata({"category": "Geometry", "difficulty": "Level 1"})

        self.assertEqual(metadata["topic"], "Geometry")
        self.assertEqual(metadata["level"], "Level 1")

    def test_build_knowledge_response_context_omits_leaked_knowledge(self):
        question = "If 120 is reduced to 96, what is the percent decrease?"
        leaked_knowledge = (
            "1. Two-Digit Number Problem: What positive two-digit integer is exactly twice the sum of its digits?\n"
            "Relevant Concepts:\n"
            "- Represent a two-digit number as 10a + b.\n"
            "2. Choosing President, Vice-President, and Treasurer from a Group of 4 Guys and 4 Girls\n"
            "Relevant Concepts:\n"
            "- Use permutations.\n"
        )

        self.assertEqual(build_knowledge_response_context(question, leaked_knowledge), "")

    def test_build_wolfram_response_context_reconstructs_trace_for_solution_prompt(self):
        context = build_wolfram_response_context(
            [
                {"step": 1, "thought": "Solve for x and y.", "query": "Solve[{...},{x,y}]", "output": "x = -23 and y = 25"},
                {"step": 2, "thought": "Compute x-y.", "query": "-23 - 25", "output": "-48"},
            ],
            "-48",
        )

        self.assertIn("Wolfram Step 1 Thought: Solve for x and y.", context)
        self.assertIn("Wolfram Step 2 Output: -48", context)
        self.assertIn("Wolfram Final Answer: -48", context)

    def test_rebuild_cached_response_uses_selected_same_question_modules_only(self):
        cache_row = {
            "dataset": "MATH",
            "example": {"type": "Geometry", "level": "Level 3"},
            "response": "Old unrelated accumulated response that should not be reused directly.",
            "knowledge_retrieval:output": "Use similar triangles.",
            "program": "print(7)",
            "program_executor:output": "7",
            "solution_generator:output": "#### 7",
        }

        rebuilt = rebuild_cached_response(cache_row, modules=["knowledge_retrieval"])

        self.assertIn("Mathematics Problem Type:Geometry", rebuilt)
        self.assertIn("Knowledge Retrieval:\nUse similar triangles.", rebuilt)
        self.assertNotIn("Old unrelated accumulated response", rebuilt)
        self.assertNotIn("Python generator:", rebuilt)
        self.assertNotIn("Solution:\n#### 7", rebuilt)

    def test_rebuild_cached_response_omits_leaked_knowledge_output(self):
        cache_row = {
            "dataset": "MATH",
            "example": {
                "type": "Algebra",
                "level": "Level 2",
                "problem": "If 120 is reduced to 96, what is the percent decrease?",
            },
            "knowledge_retrieval:output": (
                "1. Two-Digit Number Problem: What positive two-digit integer is exactly twice the sum of its digits?\n"
                "Relevant Concepts:\n"
                "- Represent a two-digit number as 10a + b.\n"
                "2. Choosing President, Vice-President, and Treasurer from a Group of 4 Guys and 4 Girls\n"
                "Relevant Concepts:\n"
                "- Use permutations.\n"
            ),
        }

        rebuilt = rebuild_cached_response(cache_row, modules=["knowledge_retrieval"])

        self.assertIn("Mathematics Problem Type:Algebra", rebuilt)
        self.assertNotIn("Two-Digit Number Problem", rebuilt)
        self.assertNotIn("Knowledge Retrieval:\n", rebuilt)


if __name__ == "__main__":
    unittest.main()

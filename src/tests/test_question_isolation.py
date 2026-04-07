import os
import sys
import unittest


TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(TESTS_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from core.question_isolation import (
    build_math_knowledge_demo_prompt,
    build_option_program_demo_prompt,
    build_option_solution_demo_prompt,
    build_option_wolfram_demo_prompt,
    cross_problem_leak_details,
    program_uses_hidden_geometry_coordinates,
    question_explicitly_uses_coordinates,
    looks_like_knowledge_leak,
    looks_like_cross_problem_leak,
)


class TestQuestionIsolation(unittest.TestCase):
    def test_math_knowledge_prompt_does_not_embed_legacy_examples(self):
        prompt = build_math_knowledge_demo_prompt()

        self.assertNotIn("What positive two-digit integer is exactly twice the sum of its digits?", prompt)
        self.assertNotIn("In how many ways can a President", prompt)
        self.assertIn("Never echo prompt examples", prompt)

    def test_option_prompts_do_not_embed_legacy_aqua_example(self):
        prompts = [
            build_option_program_demo_prompt("A-E"),
            build_option_solution_demo_prompt("A-E"),
            build_option_wolfram_demo_prompt("A-E"),
        ]

        for prompt in prompts:
            self.assertNotIn("If a / b = 3/4", prompt)
            self.assertNotIn("8a + 5b = 22", prompt)

    def test_option_prompts_require_reverse_check_for_estimates(self):
        self.assertIn("compare it against every option", build_option_program_demo_prompt("A-E"))
        self.assertIn("do not round intermediate quantities", build_option_program_demo_prompt("A-E"))
        self.assertIn("do not round first and compare later", build_option_program_demo_prompt("A-E"))
        self.assertIn("each option before printing the final letter", build_option_program_demo_prompt("A-E"))
        self.assertIn("reverse-check the final choice", build_option_solution_demo_prompt("A-E"))
        self.assertIn('Final Answer: [LETTER]. [OPTION TEXT]', build_option_solution_demo_prompt("A-E"))
        self.assertIn("both the option letter and its value", build_option_solution_demo_prompt("A-E"))
        self.assertIn("actual target quantity first", build_option_wolfram_demo_prompt("A-E"))

    def test_geometry_prompts_forbid_hidden_coordinates(self):
        self.assertIn("do not assign coordinates", build_option_program_demo_prompt("A-E"))
        self.assertIn("hidden diagram coordinates", build_option_solution_demo_prompt("A-E"))
        self.assertIn("invented coordinates", build_option_wolfram_demo_prompt("A-E"))
        self.assertIn("explicitly gives coordinates", build_math_knowledge_demo_prompt())

    def test_question_coordinate_detection_distinguishes_explicit_coordinate_geometry(self):
        self.assertTrue(question_explicitly_uses_coordinates("Points A(0, 0) and B(3, 4) lie on the plane."))
        self.assertFalse(question_explicitly_uses_coordinates("In triangle ABC, AB = AC and angle B = 40 degrees."))

    def test_geometry_program_detector_flags_invented_coordinates_without_explicit_coordinate_prompt(self):
        question = "In triangle ABC, AB = AC and angle B = 40 degrees. Find angle C."
        program = (
            "from sympy import *\n"
            "A = (0, 0)\n"
            "B = (1, 0)\n"
            "C = (cos(pi/4), sin(pi/4))\n"
            "print(C)\n"
        )

        self.assertTrue(program_uses_hidden_geometry_coordinates(question, program))

    def test_geometry_program_detector_allows_coordinate_methods_when_problem_explicitly_gives_coordinates(self):
        question = "Given points A(0, 0), B(3, 0), and C(0, 4), find the area of triangle ABC."
        program = (
            "from sympy import *\n"
            "A = (0, 0)\n"
            "B = (3, 0)\n"
            "C = (0, 4)\n"
            "print(6)\n"
        )

        self.assertFalse(program_uses_hidden_geometry_coordinates(question, program))

    def test_cross_problem_leak_detector_flags_unrelated_aqua_program(self):
        question = "If 120 is reduced to 96, what is the reduction percent?"
        leaked_program = (
            "from sympy import *\n"
            "a, b = symbols('a b')\n"
            "equation1 = Eq(a / b, 3 / 4)\n"
            "equation2 = Eq(8 * a + 5 * b, 22)\n"
            "solutions = solve((equation1, equation2), (a, b), dict=True)\n"
            "print(solutions[0][a])\n"
        )

        self.assertTrue(looks_like_cross_problem_leak(question, leaked_program, mode="program"))
        details = cross_problem_leak_details(question, leaked_program)
        self.assertIn("22", details["foreign_numbers"])
        self.assertEqual(details["overlap_numbers"], [])

    def test_cross_problem_leak_detector_allows_grounded_aqua_program(self):
        question = "If 120 is reduced to 96, what is the reduction percent?"
        grounded_program = (
            "from sympy import *\n"
            "original = 120\n"
            "new_value = 96\n"
            "reduction = original - new_value\n"
            "percent = reduction * 100 / original\n"
            "print(percent)\n"
        )

        self.assertFalse(looks_like_cross_problem_leak(question, grounded_program, mode="program"))

    def test_knowledge_leak_detector_flags_multi_problem_prompt_echo(self):
        question = "If 120 is reduced to 96, what is the percent decrease?"
        leaked_knowledge = (
            "Certainly! Here is the relevant background knowledge and mathematical concepts that would be helpful "
            "for understanding and solving each of the given problems:\n\n"
            "---\n\n"
            "1. Two-Digit Number Problem: What positive two-digit integer is exactly twice the sum of its digits?\n"
            "Relevant Concepts:\n"
            "- Represent a two-digit number as 10a + b.\n\n"
            "---\n\n"
            "2. Choosing President, Vice-President, and Treasurer from a Group of 4 Guys and 4 Girls\n"
            "Relevant Concepts:\n"
            "- Use permutations and complement counting.\n"
        )

        self.assertTrue(looks_like_knowledge_leak(question, leaked_knowledge))

    def test_knowledge_leak_detector_allows_grounded_single_question_knowledge(self):
        question = "If 120 is reduced to 96, what is the percent decrease?"
        grounded_knowledge = (
            "- Percent decrease equals (original - new) / original times 100.\n"
            "- Here the original value is 120 and the new value is 96.\n"
            "- Compute the decrease first, then convert it to a percent."
        )

        self.assertFalse(looks_like_knowledge_leak(question, grounded_knowledge))


if __name__ == "__main__":
    unittest.main()

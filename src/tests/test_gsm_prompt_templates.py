import os
import sys
import unittest


TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(TESTS_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from core.gsm_prompt_templates import (
    build_gsm_knowledge_demo_prompt,
    build_gsm_program_demo_prompt,
    build_gsm_solution_demo_prompt,
    build_gsm_wolfram_demo_prompt,
    gsm_story_leak_tokens,
    looks_like_gsm_story_leak,
)


class TestGsmPromptTemplates(unittest.TestCase):
    def test_clean_gsm_prompts_do_not_embed_old_story_examples(self):
        prompts = [
            build_gsm_knowledge_demo_prompt(),
            build_gsm_program_demo_prompt(),
            build_gsm_solution_demo_prompt(),
            build_gsm_wolfram_demo_prompt(),
        ]

        for prompt in prompts:
            self.assertNotIn("Tim has some cans of soda", prompt)
            self.assertNotIn("initial_cans", prompt)

    def test_gsm_wolfram_prompt_requires_parenthesized_unit_free_arithmetic(self):
        prompt = build_gsm_wolfram_demo_prompt()

        self.assertIn("plain arithmetic with numbers and operators only", prompt)
        self.assertIn("Do not include unit words", prompt)
        self.assertIn("Use explicit parentheses", prompt)

    def test_story_leak_detector_flags_unrelated_gsm_code(self):
        question = (
            "Travis had 61 apps on his tablet. He deleted 9 apps he did not use anymore and "
            "downloaded 18 more. How many apps are on his tablet now?"
        )
        leaked_code = (
            "from sympy import *\n"
            "initial_cans = symbols('initial_cans')\n"
            "cans_left_after_jeff = initial_cans - 6\n"
            "cans_after_purchase = cans_left_after_jeff + Rational(1, 2) * cans_left_after_jeff\n"
            "print(22)\n"
        )

        self.assertTrue(looks_like_gsm_story_leak(question, leaked_code))
        self.assertIn("cans", gsm_story_leak_tokens(question, leaked_code))
        self.assertIn("jeff", gsm_story_leak_tokens(question, leaked_code))

    def test_story_leak_detector_allows_question_grounded_gsm_code(self):
        question = (
            "Travis had 61 apps on his tablet. He deleted 9 apps he did not use anymore and "
            "downloaded 18 more. How many apps are on his tablet now?"
        )
        grounded_code = (
            "from sympy import *\n"
            "# Track Travis's apps after deleting and downloading.\n"
            "starting_apps = 61\n"
            "apps_after_delete = starting_apps - 9\n"
            "apps_now = apps_after_delete + 18\n"
            "print(apps_now)\n"
        )

        self.assertFalse(looks_like_gsm_story_leak(question, grounded_code))


if __name__ == "__main__":
    unittest.main()

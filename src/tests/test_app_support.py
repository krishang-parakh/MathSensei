import os
import sys
import unittest
from types import SimpleNamespace


TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(TESTS_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from core.app_support import apply_global_model_overrides, solution_prompt_family


class TestAppSupport(unittest.TestCase):
    def test_global_model_sets_all_module_backends(self):
        args = SimpleNamespace(
            global_model="gemini",
            python_model="no",
            knowledge_model="no",
            bing_model="no",
            sg_model="no",
            wolfram_model="no",
        )

        updated = apply_global_model_overrides(args)

        self.assertEqual(updated.python_model, "gemini")
        self.assertEqual(updated.knowledge_model, "gemini")
        self.assertEqual(updated.bing_model, "gemini")
        self.assertEqual(updated.sg_model, "gemini")
        self.assertEqual(updated.wolfram_model, "gemini")

    def test_specific_module_settings_override_global_model(self):
        args = SimpleNamespace(
            global_model="gemini",
            python_model="no",
            knowledge_model="text_davinci_003",
            bing_model="no",
            sg_model="gemini",
            wolfram_model="no",
        )

        updated = apply_global_model_overrides(args)

        self.assertEqual(updated.python_model, "gemini")
        self.assertEqual(updated.knowledge_model, "text_davinci_003")
        self.assertEqual(updated.bing_model, "gemini")
        self.assertEqual(updated.sg_model, "gemini")
        self.assertEqual(updated.wolfram_model, "gemini")

    def test_routes_kr_solution_models_to_their_own_prompt_families(self):
        self.assertEqual(solution_prompt_family("kr_sg"), "kr_sg")
        self.assertEqual(solution_prompt_family("kr_pg_sg"), "kr_pg_sg")
        self.assertEqual(solution_prompt_family("kr_walpha_sg"), "kr_walpha_sg")
        self.assertEqual(solution_prompt_family("kr_pg_walpha_sg"), "kr_walpha_sg")

    def test_routes_cot_and_pot_separately(self):
        self.assertEqual(solution_prompt_family("cot"), "cot")
        self.assertEqual(solution_prompt_family("pot"), "pot")

    def test_defaults_other_solution_models_to_kr_walpha_family(self):
        self.assertEqual(solution_prompt_family("sg"), "kr_walpha_sg")
        self.assertEqual(solution_prompt_family("pg_sg"), "kr_walpha_sg")
        self.assertEqual(solution_prompt_family("planner"), "kr_walpha_sg")


if __name__ == "__main__":
    unittest.main()

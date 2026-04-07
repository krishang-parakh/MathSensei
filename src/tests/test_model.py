import os
import sys
import unittest


TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(TESTS_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import model as model_module
from model import solver


class TestModelProgramPipeline(unittest.TestCase):
    def setUp(self):
        self._orig_get_chat_response = model_module.get_chat_response
        self._orig_get_gemini_response = model_module.get_gemini_response

    def tearDown(self):
        model_module.get_chat_response = self._orig_get_chat_response
        model_module.get_gemini_response = self._orig_get_gemini_response

    def test_get_metadata_handles_examples_without_subject_type_or_level(self):
        instance = solver.__new__(solver)
        instance.cache = {
            "example": {
                "problem": "I have 2 faces, no vertices, and I can roll.",
            }
        }

        metadata = solver.get_metadata(instance)

        self.assertEqual(metadata["topic"], "")
        self.assertEqual(metadata["level"], "")

    def test_program_executor_recovers_by_regenerating_missing_program(self):
        instance = solver.__new__(solver)
        instance.cache = {
            "response": "",
            "example": {"problem": "What is 2 + 3?"},
        }
        instance.modules = ["program_generator", "program_executor", "solution_generator"]
        instance.dataset = "GSM"
        instance.dependency_install_attempts = set()
        generator_calls = []

        def fake_program_generator():
            generator_calls.append("program_generator")
            code = "from sympy import *\nprint(5)"
            instance.cache["program"] = code
            instance.cache["program_generator:output"] = code
            return "prompt", code

        def fake_attempt_program_repairs(program):
            return program, "5", None, None

        instance.program_generator = fake_program_generator
        instance._attempt_program_repairs = fake_attempt_program_repairs

        program, output = solver.program_executor(instance)

        self.assertEqual(generator_calls, ["program_generator"])
        self.assertEqual(program.strip(), "from sympy import *\nprint(5)")
        self.assertEqual(output, "5")
        self.assertEqual(instance.cache["program_executor:output"], "5")
        self.assertIsNone(instance.cache["program_executor:error"])

    def test_python_generator_falls_back_to_default_chat_after_blank_noncode_backend_reply(self):
        instance = solver.__new__(solver)
        instance.cache = {"example": {"problem": "What is 2 + 3?"}}
        instance.python_model = "gemini"
        instance.pg_temperature = 0
        instance.pg_max_tokens = 200

        model_module.get_gemini_response = lambda prompt: "I think the answer is 5."
        model_module.get_chat_response = lambda messages, temperature, max_tokens: "from sympy import *\nprint(5)"

        program = solver._generate_python_program(instance, "Question: What is 2 + 3?\nCode:\n")

        self.assertEqual(program.strip(), "from sympy import *\nprint(5)")
        attempts = instance.cache["program_generator:attempts"]
        self.assertEqual(attempts[0]["backend"], "gemini")
        self.assertFalse(attempts[0]["usable"])
        self.assertEqual(attempts[-1]["backend"], "default-chat")
        self.assertTrue(attempts[-1]["usable"])

    def test_retry_python_program_for_repo_echo_regenerates_standalone_code(self):
        instance = solver.__new__(solver)
        instance.cache = {}
        instance.python_model = "no"
        instance.pg_temperature = 0
        instance.pg_max_tokens = 200
        instance._current_options = lambda: None
        responses = iter(
            [
                "from sympy import *\nprint(7)",
            ]
        )
        instance._generate_python_program = lambda prompt: next(responses)

        repaired = solver._retry_python_program_for_repo_echo(
            instance,
            "Question: compute 7\nCode:\n",
            "compute 7",
            "import os\nclass solver:\n    pass",
        )

        self.assertEqual(repaired.strip(), "from sympy import *\nprint(7)")
        self.assertIn(
            "program_generator: repo-echo draft was regenerated into standalone code.",
            instance.cache.get("module_warnings", []),
        )

    def test_last_resort_python_program_uses_first_option_letter_for_multiple_choice(self):
        instance = solver.__new__(solver)
        instance.cache = {"example": {"options": [{"key": "C", "label": "200 + 450 + 200"}]}}
        instance.dataset = "AQUA"

        program = solver._build_last_resort_python_program(instance)

        self.assertIn("print('C')", program)


if __name__ == "__main__":
    unittest.main()

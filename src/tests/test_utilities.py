import os
import sys
import types
import unittest
from unittest import mock


TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(TESTS_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

os.environ.setdefault("OPENAI_API_BASE", "https://example.invalid")
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("OPENAI_API_VERSION", "2024-12-01-preview")
os.environ.setdefault("OPENAI_DEPLOYMENT_NAME", "test-deployment")

if "matplotlib" not in sys.modules:
    matplotlib_stub = types.ModuleType("matplotlib")
    matplotlib_stub.use = lambda *args, **kwargs: None
    sys.modules["matplotlib"] = matplotlib_stub

if "google.generativeai" not in sys.modules:
    google_module = sys.modules.setdefault("google", types.ModuleType("google"))
    generativeai_stub = types.ModuleType("google.generativeai")
    generativeai_stub.configure = lambda *args, **kwargs: None
    sys.modules["google.generativeai"] = generativeai_stub
    setattr(google_module, "generativeai", generativeai_stub)

if "dotenv" not in sys.modules:
    dotenv_stub = types.ModuleType("dotenv")
    dotenv_stub.load_dotenv = lambda *args, **kwargs: None
    sys.modules["dotenv"] = dotenv_stub

if "openai" not in sys.modules:
    openai_stub = types.ModuleType("openai")

    class _AzureOpenAIStub:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    openai_stub.AzureOpenAI = _AzureOpenAIStub
    sys.modules["openai"] = openai_stub

from utilities import (
    _candidate_gemini_model_names,
    _http_status_code_from_error,
    _sanitize_messages_for_model_input,
    _sanitize_text_for_model_input,
    _split_gemini_model_names,
    safe_execute,
)
from presentation.asy_rendering import MODEL_INPUT_DIAGRAM_NOTICE


class TestUtilities(unittest.TestCase):
    def test_split_gemini_model_names_supports_lists(self):
        raw_value = "gemini-2.5-pro, gemini-2.5-flash;\nmodels/gemini-2.0-flash"

        self.assertEqual(
            _split_gemini_model_names(raw_value),
            ["gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.0-flash"],
        )

    def test_candidate_models_keep_flash_close_to_preferred_pro(self):
        candidates = _candidate_gemini_model_names(
            preferred="gemini-2.5-pro",
            discovered=["gemini-2.0-flash-lite"],
        )

        self.assertEqual(candidates[0], "gemini-2.5-pro")
        self.assertIn("gemini-2.5-flash", candidates[:3])
        self.assertIn("gemini-2.0-flash", candidates[:4])
        self.assertIn("gemini-2.0-flash-lite", candidates)

    def test_http_status_code_can_be_parsed_from_error_text(self):
        error = RuntimeError("429 Client Error: Too Many Requests for url: https://example.com")
        self.assertEqual(_http_status_code_from_error(error), 429)

    def test_safe_execute_includes_exception_type_for_keyerror(self):
        output, error = safe_execute("values = {1: 2}\nprint(values[0])")

        self.assertIsNone(output)
        self.assertIn("KeyError", error)
        self.assertIn("0", error)

    def test_safe_execute_allows_evalf_on_plain_python_numbers(self):
        output, error = safe_execute("value = 20.0\nprint(value.evalf())")

        self.assertIsNone(error)
        self.assertEqual(output.strip(), "20.0")

    def test_safe_execute_preserves_sympy_evalf_for_sympy_objects(self):
        output, error = safe_execute("from sympy import *\nvalue = Rational(1, 3)\nprint(value.evalf(5))")

        self.assertIsNone(error)
        self.assertIn("0.33333", output)

    def test_safe_execute_retries_after_installing_missing_dependency(self):
        module_name = "autoinstall_test_module"
        sys.modules.pop(module_name, None)

        def fake_install(error_message):
            module = types.ModuleType(module_name)
            module.VALUE = 7
            sys.modules[module_name] = module
            return True, {
                "module": module_name,
                "package": module_name,
                "installed": True,
            }

        with mock.patch("utilities._attempt_missing_dependency_install", side_effect=fake_install) as patched:
            output, error = safe_execute(f"import {module_name}\nprint({module_name}.VALUE)")

        self.assertEqual(patched.call_count, 1)
        self.assertIsNone(error)
        self.assertEqual(output.strip(), "7")
        sys.modules.pop(module_name, None)

    def test_sanitize_text_for_model_input_replaces_asy_with_policy_notice(self):
        prompt = "Question: Find x.\n[asy]\ndraw((0,0)--(1,1));\n[/asy]\nAnswer:"

        sanitized = _sanitize_text_for_model_input(prompt)

        self.assertIn(MODEL_INPUT_DIAGRAM_NOTICE, sanitized)
        self.assertNotIn("draw((0,0)--(1,1));", sanitized)

    def test_sanitize_messages_for_model_input_replaces_asy_in_message_content(self):
        messages = [
            {"role": "system", "content": "Use rigorous geometry."},
            {"role": "user", "content": "Question:\n[asy]\ndot((0,0));\n[/asy]\nSolve it."},
        ]

        sanitized = _sanitize_messages_for_model_input(messages)

        self.assertEqual(sanitized[0]["content"], "Use rigorous geometry.")
        self.assertIn(MODEL_INPUT_DIAGRAM_NOTICE, sanitized[1]["content"])
        self.assertNotIn("dot((0,0));", sanitized[1]["content"])


if __name__ == "__main__":
    unittest.main()

import os
import sys
import tempfile
import types
import unittest
from unittest import mock


TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(TESTS_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

for env_name in ("OPENAI_PROVIDER", "OPENAI_API_BASE", "OPENAI_API_KEY", "OPENAI_API_VERSION", "OPENAI_DEPLOYMENT_NAME", "MODEL_NAME"):
    os.environ.pop(env_name, None)

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

    class _OpenAIStub:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    openai_stub.AzureOpenAI = _AzureOpenAIStub
    openai_stub.OpenAI = _OpenAIStub
    sys.modules["openai"] = openai_stub

from utilities import (
    _chat_completion,
    _chat_completion_azure,
    _build_reasoning_fallback_kwargs,
    _candidate_gemini_model_names,
    _diagnose_openai_error,
    _extract_supported_reasoning_efforts,
    _http_status_code_from_error,
    _looks_like_broken_loopback_proxy,
    _sanitize_messages_for_model_input,
    _sanitize_text_for_model_input,
    _split_gemini_model_names,
    _standard_openai_reasoning_kwargs,
    safe_execute,
)
from core import env_loader
from presentation.asy_rendering import MODEL_INPUT_DIAGRAM_NOTICE


class TestUtilities(unittest.TestCase):
    def test_utilities_import_does_not_require_fake_azure_env(self):
        self.assertTrue(callable(_chat_completion))

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

    def test_broken_loopback_proxy_detection_flags_discard_port(self):
        self.assertTrue(_looks_like_broken_loopback_proxy("http://127.0.0.1:9"))
        self.assertTrue(_looks_like_broken_loopback_proxy("http://localhost:9"))
        self.assertFalse(_looks_like_broken_loopback_proxy("http://127.0.0.1:8080"))

    def test_openai_error_diagnostic_highlights_404_config_mismatch(self):
        diagnostic = _diagnose_openai_error(RuntimeError("Error code: 404 - {'error': {'code': '404', 'message': 'Resource not found'}}"))

        self.assertIn("Resource not found", diagnostic)
        self.assertIn("OPENAI_DEPLOYMENT_NAME", diagnostic)
        self.assertIn("OPENAI_API_BASE", diagnostic)

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

    def test_chat_completion_uses_standard_openai_responses_api_when_sk_key_is_present(self):
        response_api = mock.Mock()
        response_api.create.return_value = types.SimpleNamespace(output_text="OK<STOP>extra")
        fake_client = types.SimpleNamespace(responses=response_api)

        with mock.patch("utilities._chat_backend_mode", return_value="openai"), mock.patch(
            "utilities._get_standard_openai_client", return_value=fake_client
        ), mock.patch("utilities.MODEL_NAME", "model-router"):
            output = _chat_completion(
                messages=[{"role": "system", "content": "Reply briefly."}, {"role": "user", "content": "Say OK"}],
                temperature=0.3,
                max_tokens=2000,
                stop=["<STOP>"],
            )

        self.assertEqual(output, "OK")
        response_api.create.assert_called_once()
        create_kwargs = response_api.create.call_args.kwargs
        self.assertEqual(create_kwargs["model"], "model-router")
        self.assertEqual(create_kwargs["max_output_tokens"], 2000)
        self.assertEqual(create_kwargs["input"][0]["role"], "system")

    def test_standard_openai_reasoning_defaults_to_low_for_gpt5_family(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            kwargs = _standard_openai_reasoning_kwargs("gpt-5.4-nano")

        self.assertEqual(kwargs, {"reasoning": {"effort": "low"}})

    def test_extract_supported_reasoning_efforts_parses_error_payload(self):
        values = _extract_supported_reasoning_efforts(
            "Unsupported value: 'minimal'. Supported values are: 'none', 'low', 'medium', 'high', and 'xhigh'."
        )

        self.assertEqual(values, ["none", "low", "medium", "high", "xhigh"])

    def test_build_reasoning_fallback_kwargs_removes_unsupported_effort(self):
        base = {
            "model": "gpt-5.4-nano",
            "input": [{"role": "user", "content": [{"type": "input_text", "text": "hi"}]}],
            "max_output_tokens": 100,
            "reasoning": {"effort": "minimal"},
        }
        fallback_kwargs = _build_reasoning_fallback_kwargs(
            base,
            "Supported values are: 'none', 'low', 'medium', 'high', and 'xhigh'.",
        )

        efforts = [kwargs.get("reasoning", {}).get("effort") for kwargs in fallback_kwargs if kwargs.get("reasoning")]
        self.assertIn("low", efforts)
        self.assertNotIn("minimal", efforts)

    def test_chat_completion_falls_back_when_requested_model_does_not_exist(self):
        response_api = mock.Mock()
        response_api.create.side_effect = [
            RuntimeError("Error code: 400 - {'error': {'message': \"The requested model 'model-router' does not exist.\", 'code': 'model_not_found'}}"),
            types.SimpleNamespace(output_text="Recovered"),
        ]
        fake_client = types.SimpleNamespace(responses=response_api)

        with mock.patch("utilities._chat_backend_mode", return_value="openai"), mock.patch(
            "utilities._get_standard_openai_client", return_value=fake_client
        ), mock.patch("utilities.MODEL_NAME", "model-router"):
            output = _chat_completion(
                messages=[{"role": "user", "content": "Say hi"}],
                temperature=0.3,
                max_tokens=2000,
            )

        self.assertEqual(output, "Recovered")
        self.assertEqual(response_api.create.call_args_list[0].kwargs["model"], "model-router")
        self.assertEqual(response_api.create.call_args_list[1].kwargs["model"], "model-router")

    def test_chat_completion_azure_prefers_max_completion_tokens(self):
        create_api = mock.Mock()
        create_api.create.return_value = types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=types.SimpleNamespace(content="Azure OK"))]
        )
        fake_client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=create_api))

        with mock.patch("utilities._get_azure_client", return_value=fake_client), mock.patch(
            "utilities.AZURE_DEPLOYMENT_NAME", "pandu"
        ):
            output = _chat_completion_azure(
                messages=[{"role": "user", "content": "Reply with exactly OK"}],
                temperature=0.2,
                max_tokens=2000,
            )

        self.assertEqual(output, "Azure OK")
        create_kwargs = create_api.create.call_args.kwargs
        self.assertEqual(create_kwargs["model"], "pandu")
        self.assertEqual(create_kwargs["max_completion_tokens"], 2000)
        self.assertNotIn("max_tokens", create_kwargs)

    def test_chat_completion_azure_retries_without_temperature_when_model_rejects_it(self):
        create_api = mock.Mock()
        create_api.create.side_effect = [
            RuntimeError(
                "Error code: 400 - {'error': {'message': \"Unsupported value: 'temperature' does not support 0.5 with this model. Only the default (1) value is supported.\", 'param': 'temperature', 'code': 'unsupported_value'}}"
            ),
            types.SimpleNamespace(
                choices=[types.SimpleNamespace(message=types.SimpleNamespace(content="Azure fallback OK"))]
            ),
        ]
        fake_client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=create_api))

        with mock.patch("utilities._get_azure_client", return_value=fake_client), mock.patch(
            "utilities.AZURE_DEPLOYMENT_NAME", "pandu"
        ):
            output = _chat_completion_azure(
                messages=[{"role": "user", "content": "Reply with exactly OK"}],
                temperature=0.5,
                max_tokens=2000,
            )

        self.assertEqual(output, "Azure fallback OK")
        first_kwargs = create_api.create.call_args_list[0].kwargs
        second_kwargs = create_api.create.call_args_list[1].kwargs
        self.assertIn("temperature", first_kwargs)
        self.assertNotIn("temperature", second_kwargs)
        self.assertEqual(second_kwargs["max_completion_tokens"], 2000)

    def test_env_loader_fallback_parses_dotenv_without_python_dotenv(self):
        with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8") as handle:
            handle.write('TEST_ENV_LOADER_KEY = "loaded-value"\n')
            dotenv_path = handle.name

        os.environ.pop("TEST_ENV_LOADER_KEY", None)
        try:
            with mock.patch.object(env_loader, "_dotenv_load_dotenv", None):
                loaded = env_loader.load_dotenv(dotenv_path=dotenv_path, override=True)

            self.assertTrue(loaded)
            self.assertEqual(os.environ.get("TEST_ENV_LOADER_KEY"), "loaded-value")
        finally:
            os.environ.pop("TEST_ENV_LOADER_KEY", None)
            os.remove(dotenv_path)


if __name__ == "__main__":
    unittest.main()

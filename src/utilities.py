import ast
import os
import time
import logging
import codecs
import re
from io import StringIO
from contextlib import redirect_stdout
from math import isclose
from typing import Any, Optional
from urllib.parse import urlparse

try:
    import matplotlib
except Exception:
    matplotlib = None
import httpx
import requests
try:
    import google.generativeai as genai
except Exception:
    genai = None
from presentation.asy_rendering import strip_asy_blocks_for_model_input
try:
    from dotenv import load_dotenv
except Exception:
    def load_dotenv(*args, **kwargs):
        return False
try:
    from openai import AzureOpenAI, OpenAI
except Exception:
    AzureOpenAI = None
    OpenAI = None

logging.basicConfig(
    format="%(asctime)s %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
    level=logging.INFO,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(BASE_DIR, ".env")
_PRE_DOTENV_OPENAI_ENV = {
    name: os.environ.get(name)
    for name in (
        "OPENAI_API_KEY",
        "OPENAI_API_BASE",
        "OPENAI_API_VERSION",
        "OPENAI_DEPLOYMENT_NAME",
        "MODEL_NAME",
    )
}
load_dotenv(ENV_PATH, override=True)

if matplotlib is not None:
    matplotlib.use("Agg")


def _get_env(name: str, default: Optional[str] = None, required: bool = False) -> Optional[str]:
    value = os.getenv(name, default)
    if required and (value is None or str(value).strip() == ""):
        raise RuntimeError(f"Missing required environment variable: {name}. Check {ENV_PATH}")
    return value


def _clean_env_value(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    value = str(value).strip()
    return value or None


def _looks_like_standard_openai_key(value: Optional[str]) -> bool:
    return bool(_clean_env_value(value) and _clean_env_value(value).startswith("sk-"))


_PROXY_ENV_NAMES = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
)


def _looks_like_broken_loopback_proxy(proxy_url: Optional[str]) -> bool:
    if proxy_url in (None, ""):
        return False

    try:
        parsed = urlparse(str(proxy_url).strip())
    except Exception:
        return False

    host = (parsed.hostname or "").strip().lower()
    port = parsed.port
    if host not in {"127.0.0.1", "localhost", "::1", "0.0.0.0"}:
        return False
    return port == 9


def _should_bypass_env_proxies() -> bool:
    return any(_looks_like_broken_loopback_proxy(os.getenv(name)) for name in _PROXY_ENV_NAMES)


PROXY_BYPASS_ACTIVE = _should_bypass_env_proxies()
if PROXY_BYPASS_ACTIVE:
    logging.warning(
        "Detected broken proxy environment variables pointing to a loopback discard port; "
        "MathSensei will bypass environment proxies for outbound API requests."
    )


def _new_requests_session():
    session = requests.Session()
    session.trust_env = not PROXY_BYPASS_ACTIVE
    return session


def _requests_request(method: str, url: str, **kwargs):
    session = _new_requests_session()
    try:
        return session.request(method=method.upper(), url=url, **kwargs)
    finally:
        session.close()


_ENV_OPENAI_API_KEY = _clean_env_value(_get_env("OPENAI_API_KEY"))
_SHELL_OPENAI_API_KEY = _clean_env_value(_PRE_DOTENV_OPENAI_ENV.get("OPENAI_API_KEY"))

STANDARD_OPENAI_API_KEY = None
if _looks_like_standard_openai_key(_SHELL_OPENAI_API_KEY):
    STANDARD_OPENAI_API_KEY = _SHELL_OPENAI_API_KEY
elif _looks_like_standard_openai_key(_ENV_OPENAI_API_KEY):
    STANDARD_OPENAI_API_KEY = _ENV_OPENAI_API_KEY

AZURE_ENDPOINT = _clean_env_value(_get_env("OPENAI_API_BASE"))
AZURE_API_KEY = _ENV_OPENAI_API_KEY if not _looks_like_standard_openai_key(_ENV_OPENAI_API_KEY) else None
AZURE_API_VERSION = _clean_env_value(_get_env("OPENAI_API_VERSION"))
AZURE_DEPLOYMENT_NAME = _clean_env_value(_get_env("OPENAI_DEPLOYMENT_NAME"))
MODEL_NAME = (
    _clean_env_value(_PRE_DOTENV_OPENAI_ENV.get("MODEL_NAME"))
    or _clean_env_value(_get_env("MODEL_NAME"))
    or AZURE_DEPLOYMENT_NAME
    or _clean_env_value(_get_env("DEFAULT_ENGINE"))
    or "gpt-5-nano"
)

if STANDARD_OPENAI_API_KEY and AZURE_ENDPOINT and AZURE_DEPLOYMENT_NAME:
    logging.info(
        "Detected a standard OpenAI project key from the shell plus Azure config in src/.env; "
        "MathSensei will prefer the standard OpenAI API for chat calls using model %r.",
        MODEL_NAME,
    )
elif _looks_like_standard_openai_key(_ENV_OPENAI_API_KEY) and AZURE_ENDPOINT:
    logging.warning(
        "OPENAI_API_KEY looks like a standard OpenAI project key (starts with 'sk-'), "
        "so chat calls will use the standard OpenAI API instead of AzureOpenAI."
    )

_azure_client = None
_standard_openai_client = None


def _new_httpx_client():
    return httpx.Client(trust_env=not PROXY_BYPASS_ACTIVE)


def _has_standard_openai_config() -> bool:
    return bool(STANDARD_OPENAI_API_KEY)


def _has_azure_chat_config() -> bool:
    return bool(AZURE_ENDPOINT and AZURE_API_KEY and AZURE_API_VERSION and AZURE_DEPLOYMENT_NAME)


def _chat_backend_mode() -> str:
    if _has_standard_openai_config():
        return "openai"
    if _has_azure_chat_config():
        return "azure"
    return "missing"


def _get_azure_client():
    global _azure_client
    if _azure_client is not None:
        return _azure_client
    if AzureOpenAI is None:
        raise RuntimeError("openai.AzureOpenAI is unavailable; install a recent openai package.")

    missing = [
        name
        for name, value in (
            ("OPENAI_API_BASE", AZURE_ENDPOINT),
            ("OPENAI_API_KEY", AZURE_API_KEY),
            ("OPENAI_API_VERSION", AZURE_API_VERSION),
            ("OPENAI_DEPLOYMENT_NAME", AZURE_DEPLOYMENT_NAME),
        )
        if not value
    ]
    if missing:
        raise RuntimeError(
            "Azure OpenAI chat is not configured. Missing: "
            + ", ".join(missing)
            + f". Check {ENV_PATH}"
        )

    _azure_client = AzureOpenAI(
        api_key=AZURE_API_KEY,
        api_version=AZURE_API_VERSION,
        azure_endpoint=AZURE_ENDPOINT,
        http_client=_new_httpx_client(),
    )
    return _azure_client


def _get_standard_openai_client():
    global _standard_openai_client
    if _standard_openai_client is not None:
        return _standard_openai_client
    if OpenAI is None:
        raise RuntimeError("openai.OpenAI is unavailable; install a recent openai package.")
    if not STANDARD_OPENAI_API_KEY:
        raise RuntimeError(
            "Standard OpenAI chat is not configured. Set OPENAI_API_KEY to an sk-... key "
            "or provide Azure chat settings in src/.env."
        )

    _standard_openai_client = OpenAI(
        api_key=STANDARD_OPENAI_API_KEY,
        http_client=_new_httpx_client(),
    )
    return _standard_openai_client

GOOGLE_API_KEY = _get_env("GOOGLE_API_KEY")
if GOOGLE_API_KEY and genai is not None:
    genai.configure(api_key=GOOGLE_API_KEY)

DEFAULT_GEMINI_MODEL_CANDIDATES = (
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-2.5-pro",
    "gemini-1.5-flash",
    "gemini-1.5-pro",
    "gemini-pro",
)
GEMINI_MODEL_NAME = _get_env("GOOGLE_GEMINI_MODEL", default="gemini-2.5-flash")


def _normalize_gemini_model_name(name: Optional[str]) -> Optional[str]:
    if name is None:
        return None
    normalized = str(name).strip()
    if normalized.startswith("models/"):
        normalized = normalized.split("/", 1)[1]
    return normalized or None


def _split_gemini_model_names(raw_value: Optional[str]):
    if raw_value is None:
        return []
    names = []
    for part in re.split(r"[,\n;]+", str(raw_value)):
        normalized = _normalize_gemini_model_name(part)
        if normalized:
            names.append(normalized)
    return names


def _discover_gemini_model_names():
    if genai is None or not hasattr(genai, "list_models"):
        return []

    try:
        discovered = []
        for model in genai.list_models():
            methods = getattr(model, "supported_generation_methods", None) or []
            name = _normalize_gemini_model_name(getattr(model, "name", None))
            if name and "generateContent" in methods and "gemini" in name.lower():
                discovered.append(name)
        return discovered
    except Exception as exc:
        logging.info("Gemini model discovery failed: %s", exc)
        return []


def _candidate_gemini_model_names(preferred: Optional[str] = None, discovered=None):
    seen = set()
    candidates = []

    preferred_names = _split_gemini_model_names(preferred)
    if not preferred_names:
        preferred_names = _split_gemini_model_names(GEMINI_MODEL_NAME)

    for name in preferred_names:
        if "pro" in name and "gemini-2.5-flash" not in preferred_names:
            # If a pro model is preferred, keep a lower-quota flash model close behind it.
            for fallback in ("gemini-2.5-flash", "gemini-2.0-flash"):
                if fallback not in preferred_names:
                    preferred_names.append(fallback)
            break

    for name in preferred_names + list(DEFAULT_GEMINI_MODEL_CANDIDATES) + list(discovered or []):
        normalized = _normalize_gemini_model_name(name)
        if normalized and normalized not in seen:
            seen.add(normalized)
            candidates.append(normalized)

    return candidates


def _http_status_code_from_error(exc) -> Optional[int]:
    response = getattr(exc, "response", None)
    if response is not None and getattr(response, "status_code", None):
        return int(response.status_code)

    match = re.search(r"\b([45]\d{2})\b", str(exc))
    if match:
        return int(match.group(1))
    return None


def _diagnose_openai_error(exc) -> Optional[str]:
    message = str(exc or "")
    diagnostics = []

    if PROXY_BYPASS_ACTIVE:
        diagnostics.append(
            "Detected broken proxy env vars such as HTTP_PROXY/HTTPS_PROXY pointing to 127.0.0.1:9; outbound API calls are bypassing env proxies."
        )

    if "only supported in v1/responses" in message:
        diagnostics.append(
            "The configured model only supports the Responses API. Standard OpenAI calls must use "
            "client.responses.create(...) rather than client.chat.completions.create(...)."
        )

    if "Resource not found" in message or _http_status_code_from_error(exc) == 404:
        diagnostics.append(
            "Azure returned 404 Resource not found. Check OPENAI_DEPLOYMENT_NAME, OPENAI_API_BASE, and OPENAI_API_VERSION. "
            f"Current values: deployment={AZURE_DEPLOYMENT_NAME!r}, endpoint={AZURE_ENDPOINT!r}, api_version={AZURE_API_VERSION!r}. "
            "This usually means the deployment name, endpoint, or API version is wrong rather than the API key."
        )

    cause = getattr(exc, "__cause__", None)
    cause_text = str(cause or "")
    if "127.0.0.1:9" in message or "127.0.0.1:9" in cause_text or "actively refused" in cause_text:
        diagnostics.append(
            "The connection was refused before authentication. A broken local proxy or blocked outbound route is more likely than an invalid API key."
        )

    if _chat_backend_mode() == "missing":
        diagnostics.append(
            "No usable chat backend is configured. Set OPENAI_API_KEY to a standard sk-... key for the standard OpenAI API, "
            "or set OPENAI_API_BASE, OPENAI_API_VERSION, OPENAI_DEPLOYMENT_NAME, and an Azure OPENAI_API_KEY for Azure OpenAI."
        )

    if not diagnostics:
        return None
    return " ".join(diagnostics)


def _normalize_code_block(code_string: str) -> str:
    new_code_string = codecs.decode(code_string, "unicode_escape")
    new_code_string = new_code_string.strip()

    if new_code_string.startswith("```python"):
        new_code_string = new_code_string[len("```python"):].strip()
    elif new_code_string.startswith("```"):
        new_code_string = new_code_string[len("```"):].strip()

    if new_code_string.endswith("```"):
        new_code_string = new_code_string[:-3].strip()

    return new_code_string


def _sanitize_text_for_model_input(text):
    if text is None:
        return None
    if isinstance(text, str):
        return strip_asy_blocks_for_model_input(text)
    if isinstance(text, list):
        sanitized_parts = []
        for part in text:
            if isinstance(part, dict):
                sanitized_part = dict(part)
                if isinstance(sanitized_part.get("text"), str):
                    sanitized_part["text"] = strip_asy_blocks_for_model_input(sanitized_part["text"])
                sanitized_parts.append(sanitized_part)
            else:
                sanitized_parts.append(part)
        return sanitized_parts
    return text


def _sanitize_messages_for_model_input(messages):
    sanitized_messages = []
    for message in messages:
        sanitized_message = dict(message)
        if "content" in sanitized_message:
            sanitized_message["content"] = _sanitize_text_for_model_input(sanitized_message["content"])
        sanitized_messages.append(sanitized_message)
    return sanitized_messages


_SAFE_EVALF_HELPER_NAME = "__mathsensei_safe_evalf__"


def _safe_evalf(value, *args, **kwargs):
    if hasattr(value, "evalf"):
        return value.evalf(*args, **kwargs)
    return value


def _rewrite_evalf_calls(code_string: str) -> str:
    try:
        parsed = ast.parse(code_string)
    except SyntaxError:
        return code_string

    class _EvalfGuardTransformer(ast.NodeTransformer):
        def visit_Call(self, node):
            self.generic_visit(node)
            if isinstance(node.func, ast.Attribute) and node.func.attr == "evalf":
                return ast.copy_location(
                    ast.Call(
                        func=ast.Name(id=_SAFE_EVALF_HELPER_NAME, ctx=ast.Load()),
                        args=[node.func.value] + list(node.args),
                        keywords=list(node.keywords),
                    ),
                    node,
                )
            return node

    rewritten = _EvalfGuardTransformer().visit(parsed)
    ast.fix_missing_locations(rewritten)

    if hasattr(ast, "unparse"):
        try:
            return ast.unparse(rewritten)
        except Exception:
            pass

    try:
        import astor
    except Exception:
        return code_string

    try:
        return astor.to_source(rewritten).strip()
    except Exception:
        return code_string


def _prepare_code_for_execution(code_string: str) -> str:
    normalized = _normalize_code_block(code_string)
    return _rewrite_evalf_calls(normalized)


def _attempt_missing_dependency_install(error_message: str):
    try:
        from core.python_pipeline import install_missing_dependency
    except Exception:
        return False, None

    install_result = install_missing_dependency(error_message)
    return bool(install_result.get("installed")), install_result

def safe_execute(code_string: str, keys=None):
    import codecs
    from io import StringIO
    from contextlib import redirect_stdout
    try:
        import matplotlib as _matplotlib
    except Exception:
        _matplotlib = None
    if _matplotlib is not None:
        _matplotlib.use('Agg')

    new_code_string = _prepare_code_for_execution(code_string)

    attempted_packages = set()

    while True:
        output = None
        error_message = None

        try:
            buffer = StringIO()
            exec_globals = {
                "__name__": "__main__",
                _SAFE_EVALF_HELPER_NAME: _safe_evalf,
            }
            with redirect_stdout(buffer):
                exec(new_code_string, exec_globals)
            output = buffer.getvalue()
            break
        except Exception as e:
            rendered_error = str(e).strip()
            if rendered_error:
                error_message = f"{type(e).__name__}: {rendered_error}"
            else:
                error_message = type(e).__name__

            installed, install_result = _attempt_missing_dependency_install(error_message)
            package_name = (install_result or {}).get("package")
            if installed and package_name and package_name not in attempted_packages:
                attempted_packages.add(package_name)
                continue
            break

    return output, error_message

def get_llama_response(tokenizer, pipeline, prompt, temperature=0.5):
    prompt = _sanitize_text_for_model_input(prompt)
    prompt = f"<s>[INST] {prompt.strip()} [/INST]"

    sequences = pipeline(
        prompt,
        do_sample=True,
        top_k=10,
        temperature=temperature,
        top_p=0.5,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=3000,
        return_full_text=False,
    )

    response = "".join(seq["generated_text"] for seq in sequences)
    index = response.find("Question")
    if index != -1:
        response = response[:index]

    return response.strip()


def get_llama_13bresponse(tokenizer, pipeline, prompt, temperature=0.5):
    prompt = _sanitize_text_for_model_input(prompt)
    sequences = pipeline(
        prompt,
        do_sample=True,
        top_k=10,
        temperature=temperature,
        top_p=0.5,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=3000,
        return_full_text=False,
    )

    response = "".join(seq["generated_text"] for seq in sequences)
    index = response.find("Question")
    if index != -1:
        response = response[:index]

    return response.strip()


def _sleep_if_needed(sleep_time: float) -> None:
    if sleep_time and sleep_time > 0:
        logging.info("---------Sleep starts----------")
        time.sleep(sleep_time)
        logging.info("---------Sleep ends----------")


def _content_to_text(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
            elif item is not None:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    return str(content)


def _get_attr_or_key(obj, name, default=None):
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _messages_to_responses_input(messages):
    response_input = []
    for message in messages:
        role = _get_attr_or_key(message, "role", "user") or "user"
        text = _content_to_text(_get_attr_or_key(message, "content", ""))
        if not text:
            continue
        response_input.append(
            {
                "role": role,
                "content": [{"type": "input_text", "text": text}],
            }
        )
    return response_input


def _extract_responses_text(response) -> str:
    output_text = _get_attr_or_key(response, "output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    output_items = _get_attr_or_key(response, "output", []) or []
    fragments = []
    for item in output_items:
        content_parts = _get_attr_or_key(item, "content", []) or []
        for part in content_parts:
            text = _get_attr_or_key(part, "text")
            if isinstance(text, str) and text:
                fragments.append(text)

    return "\n".join(fragments).strip()


def _apply_stop_sequences(text: str, stop) -> str:
    if not text or stop is None:
        return text

    if isinstance(stop, str):
        stop_sequences = [stop]
    else:
        stop_sequences = [sequence for sequence in stop if sequence]

    cut_positions = [text.find(sequence) for sequence in stop_sequences if text.find(sequence) != -1]
    if cut_positions:
        text = text[:min(cut_positions)]
    return text.strip()


def _chat_completion_standard_openai(messages, temperature=0.0, max_tokens=5000, stop=None):
    client = _get_standard_openai_client()
    kwargs = {
        "model": MODEL_NAME,
        "input": _messages_to_responses_input(messages),
        "max_output_tokens": max_tokens,
    }
    if temperature is not None:
        kwargs["temperature"] = temperature

    try:
        response = client.responses.create(**kwargs)
    except Exception as exc:
        if "temperature" in str(exc).lower() and "unsupported" in str(exc).lower():
            kwargs.pop("temperature", None)
            response = client.responses.create(**kwargs)
        else:
            raise

    return _apply_stop_sequences(_extract_responses_text(response), stop)


def get_textdavinci002_response(prompt, temperature, max_tokens, n=1, patience=1, sleep_time=0):
    deployment = _get_env("OPENAI_TEXTDAVC002_DEPLOYMENT_NAME", required=True)
    last_error = None
    prompt = _sanitize_text_for_model_input(prompt)

    while patience > 0:
        patience -= 1
        try:
            response = client.completions.create(
                model=deployment,
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=0.5,
            )
            prediction = response.choices[0].text.strip()
            if prediction:
                _sleep_if_needed(sleep_time)
                return prediction
        except Exception as e:
            last_error = e
            logging.exception("text-davinci-002 call failed: %s", e)
            _sleep_if_needed(sleep_time)

    raise RuntimeError(f"text-davinci-002 call failed after retries: {last_error}")


def get_textdavinci003_response(prompt, temperature, max_tokens, n=1, patience=1, sleep_time=0):
    deployment = _get_env("OPENAI_TEXTDAVC003_DEPLOYMENT_NAME", required=True)
    last_error = None
    prompt = _sanitize_text_for_model_input(prompt)

    logging.info("-------Text davinci 003 response------")
    while patience > 0:
        patience -= 1
        try:
            response = client.completions.create(
                model=deployment,
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=0.5,
            )
            prediction = response.choices[0].text.strip()
            if prediction:
                _sleep_if_needed(sleep_time)
                return prediction
        except Exception as e:
            last_error = e
            logging.exception("text-davinci-003 call failed: %s", e)
            _sleep_if_needed(sleep_time)

    raise RuntimeError(f"text-davinci-003 call failed after retries: {last_error}")


def get_gpt3_response(
    prompt,
    api_key,
    engine="text-davinci-002",
    temperature=0,
    max_tokens=5000,
    top_p=1,
    n=1,
    patience=100,
    sleep_time=0,
):
    last_error = None
    prompt = _sanitize_text_for_model_input(prompt)

    while patience > 0:
        patience -= 1
        try:
            response = client.completions.create(
                model=engine,
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
            )
            prediction = response.choices[0].text.strip()
            if prediction:
                return prediction
        except Exception as e:
            last_error = e
            logging.exception("GPT3 completion failed: %s", e)
            _sleep_if_needed(sleep_time)

    raise RuntimeError(f"GPT3 completion failed after retries: {last_error}")


def _chat_completion(messages, temperature=0.0, max_tokens=5000, stop=None):
    messages = _sanitize_messages_for_model_input(messages)
    
    # Determine which backend to use and get the appropriate client
    backend = _chat_backend_mode()
    if backend == "openai":
        client = _get_standard_openai_client()
        model_name = MODEL_NAME
    elif backend == "azure":
        client = _get_azure_client()
        model_name = AZURE_DEPLOYMENT_NAME
    else:
        raise RuntimeError(
            "Chat API is not configured. Set OPENAI_API_KEY (for standard OpenAI) or "
            "OPENAI_API_BASE, OPENAI_API_KEY, OPENAI_API_VERSION, OPENAI_DEPLOYMENT_NAME "
            "(for Azure) in src/.env"
        )
    
    kwargs = {
        "model": model_name,
        "messages": messages,
    }
    
    # Handle backend-specific parameters: Azure doesn't support temperature, only max_completion_tokens
    if backend == "azure":
        kwargs["max_completion_tokens"] = max_tokens
    else:
        kwargs["temperature"] = temperature
        kwargs["max_tokens"] = max_tokens
    
    if stop is not None:
        kwargs["stop"] = stop

    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content.strip()


def get_chat_response_code(
    context,
    temperature=0.5,
    max_tokens=5000,
    system_mess=None,
    stop=None,
    n=1,
    patience=10,
    sleep_time=0,
):
    logging.info("----Response starts-----")

    messages = []
    if system_mess is not None:
        messages.append({"role": "system", "content": system_mess})
    messages.append({"role": "user", "content": context})

    last_error = None
    while patience > 0:
        patience -= 1
        try:
            response_text = _chat_completion(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stop=stop,
            )
            logging.info("----Response ends-----")
            _sleep_if_needed(sleep_time)
            return response_text
        except Exception as e:
            last_error = e
            logging.exception("Azure chat code call failed: %s", e)
            _sleep_if_needed(sleep_time)

    diagnostic = _diagnose_openai_error(last_error)
    if diagnostic:
        raise RuntimeError(f"Azure chat code call failed after retries: {last_error}. {diagnostic}")
    raise RuntimeError(f"Azure chat code call failed after retries: {last_error}")


def get_gemini_response(full_prompt):
    if not GOOGLE_API_KEY:
        raise RuntimeError("GOOGLE_API_KEY is missing from .env")
    if genai is None:
        raise RuntimeError("google.generativeai is not installed")
    full_prompt = _sanitize_text_for_model_input(full_prompt)

    def candidate_model_names():
        for name in _candidate_gemini_model_names(discovered=_discover_gemini_model_names()):
            yield name

    def extract_text_from_response(response):
        text = getattr(response, "text", None)
        if text:
            return text

        candidates = getattr(response, "candidates", None) or []
        for candidate in candidates:
            content = getattr(candidate, "content", None)
            parts = getattr(content, "parts", None) or []
            collected = []
            for part in parts:
                part_text = getattr(part, "text", None)
                if part_text:
                    collected.append(part_text)
            if collected:
                return "\n".join(collected).strip()

        return None

    errors = []
    rate_limited_models = set()

    # Prefer the SDK when available.
    if genai is not None and hasattr(genai, "GenerativeModel"):
        for model_name in candidate_model_names():
            try:
                gemini_model = genai.GenerativeModel(model_name)
                response = gemini_model.generate_content(full_prompt)
                text = extract_text_from_response(response)
                if text:
                    return text
                errors.append(f"{model_name}: empty response")
            except Exception as exc:
                if _http_status_code_from_error(exc) == 429:
                    rate_limited_models.add(model_name)
                errors.append(f"{model_name}: {exc}")

    # Fallback to the REST API to avoid SDK-version incompatibilities.
    for model_name in candidate_model_names():
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={GOOGLE_API_KEY}"
            payload = {
                "contents": [
                    {
                        "parts": [
                            {"text": full_prompt}
                        ]
                    }
                ]
            }
            response = _requests_request("POST", url, json=payload, timeout=120)
            response.raise_for_status()
            data = response.json()

            candidates = data.get("candidates") or []
            for candidate in candidates:
                content = candidate.get("content") or {}
                parts = content.get("parts") or []
                collected = [part.get("text", "") for part in parts if part.get("text")]
                if collected:
                    return "\n".join(collected).strip()

            errors.append(f"{model_name}: empty REST response")
        except Exception as exc:
            if _http_status_code_from_error(exc) == 429:
                rate_limited_models.add(model_name)
            errors.append(f"{model_name}: {exc}")

    if rate_limited_models:
        logging.warning(
            "Gemini rate limit hit for models: %s",
            ", ".join(sorted(rate_limited_models)),
        )

    raise RuntimeError(
        "Gemini call failed. Attempts: "
        + " | ".join(errors)
        + " | Hint: set GOOGLE_GEMINI_MODEL to a currently available flash model such as gemini-2.5-flash if pro is rate-limited."
    )


def get_chat_response(messages, temperature=0, max_tokens=5000, n=1, patience=1, sleep_time=0):
    logging.info("CHATGPT CALLED")
    logging.info("----Response starts-----")

    last_error = None
    while patience > 0:
        patience -= 1
        try:
            response_text = _chat_completion(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            logging.info("----Response ends-----")
            _sleep_if_needed(sleep_time)
            return response_text
        except Exception as e:
            last_error = e
            logging.exception("Azure chat call failed: %s", e)
            _sleep_if_needed(sleep_time)

    diagnostic = _diagnose_openai_error(last_error)
    if diagnostic:
        raise RuntimeError(f"Azure chat call failed after retries: {last_error}. {diagnostic}")
    raise RuntimeError(f"Azure chat call failed after retries: {last_error}")


def floatify_ans(ans):
    if ans is None:
        return None
    elif type(ans) == dict:
        ans = list(ans.values())[0]
    elif type(ans) == bool:
        ans = ans
    elif type(ans) in [list, tuple]:
        if not ans:
            return None
        else:
            try:
                ans = float(ans[0])
            except Exception:
                ans = str(ans[0])
    else:
        try:
            ans = float(ans)
        except Exception:
            ans = str(ans)
    return ans


def score_string_similarity(str1, str2):
    if str1 == str2:
        return 2.0
    elif " " in str1 or " " in str2:
        str1_split = str1.split(" ")
        str2_split = str2.split(" ")
        overlap = list(set(str1_split) & set(str2_split))
        return len(overlap) / max(len(str1_split), len(str2_split))
    else:
        return 0.0


def _validate_server(address):
    if not address:
        raise ValueError("Must provide a valid server for search")
    if address.startswith("http://") or address.startswith("https://"):
        return address
    protocol = "http://"
    logging.info(f'No protocol provided, using "{protocol}"')
    return f"{protocol}{address}"


def call_bing_search(endpoint, bing_api_key, query, count):
    headers = {"Ocp-Apim-Subscription-Key": bing_api_key}
    params = {
        "q": query,
        "textDecorations": True,
        "textFormat": "HTML",
        "count": count,
    }

    try:
        response = _requests_request("GET", endpoint, headers=headers, params=params, timeout=30)
        logging.info("BING CALLED")
        if response.status_code == 200:
            return response.json()
        logging.warning("Bing search failed with status %s: %s", response.status_code, response.text)
    except Exception as e:
        logging.exception("Bing search request failed: %s", e)

    return None


def parse_bing_result(result):
    responses = []
    try:
        value = result["webPages"]["value"]
    except Exception:
        return responses

    try:
        for item in value:
            snippet = item["snippet"] if "snippet" in item else ""
            snippet = snippet.replace("<b>", "").replace("</b>", "").strip()
            if snippet != "":
                responses.append(snippet)
        return responses
    except Exception:
        return []


def get_webpage_content():
    from bs4 import BeautifulSoup

    url = "https://en.wikipedia.org/wiki/Ryan_Gosling"
    response = _requests_request("GET", url, timeout=30)
    soup = BeautifulSoup(response.text, "html.parser")
    text = soup.get_text()
    lines = text.splitlines()
    logging.info("%s", lines[1000:])

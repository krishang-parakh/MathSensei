import os
import time
import logging
import codecs
import re
from io import StringIO
from contextlib import redirect_stdout
from math import isclose
from typing import Any, Optional

import matplotlib
import requests
import google.generativeai as genai
from dotenv import load_dotenv
from openai import AzureOpenAI

logging.basicConfig(
    format="%(asctime)s %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
    level=logging.INFO,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(BASE_DIR, ".env")
load_dotenv(ENV_PATH)

matplotlib.use("Agg")


def _get_env(name: str, default: Optional[str] = None, required: bool = False) -> Optional[str]:
    value = os.getenv(name, default)
    if required and (value is None or str(value).strip() == ""):
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


AZURE_ENDPOINT = _get_env("OPENAI_API_BASE", required=True)
AZURE_API_KEY = _get_env("OPENAI_API_KEY", required=True)
AZURE_API_VERSION = _get_env("OPENAI_API_VERSION", required=True)
AZURE_DEPLOYMENT_NAME = _get_env("OPENAI_DEPLOYMENT_NAME", required=True)
MODEL_NAME = _get_env("MODEL_NAME", default=AZURE_DEPLOYMENT_NAME)

client = AzureOpenAI(
    api_key=AZURE_API_KEY,
    api_version=AZURE_API_VERSION,
    azure_endpoint=AZURE_ENDPOINT,
)

GOOGLE_API_KEY = _get_env("GOOGLE_API_KEY")
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)

GEMINI_MODEL_NAME = _get_env("GOOGLE_GEMINI_MODEL", default="gemini-1.5-pro")
GEMINI_FALLBACK_TO_AZURE = (_get_env("GEMINI_FALLBACK_TO_AZURE", default="true") or "true").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}


def _normalize_gemini_model_name(name: Optional[str]) -> Optional[str]:
    if not name:
        return None

    normalized = str(name).strip()
    if not normalized:
        return None

    lowered = normalized.lower().replace("models/", "").strip()
    alias_map = {
        "gemini 3.1 pro": "gemini-2.5-pro",
        "gemini 3.1 flash": "gemini-2.5-flash",
        "gemini 2.5 pro": "gemini-2.5-pro",
        "gemini 2.5 flash": "gemini-2.5-flash",
        "gemini 2.0 flash": "gemini-2.0-flash",
        "gemini 1.5 pro": "gemini-1.5-pro",
        "gemini 1.5 flash": "gemini-1.5-flash",
        "gemini pro": "gemini-pro",
        "gemini flash": "gemini-1.5-flash",
    }
    return alias_map.get(lowered, lowered)


def _list_gemini_models_via_rest():
    if not GOOGLE_API_KEY:
        return []

    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={GOOGLE_API_KEY}"
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    data = response.json()

    models = []
    for item in data.get("models", []):
        supported_methods = item.get("supportedGenerationMethods") or []
        if "generateContent" not in supported_methods:
            continue

        name = item.get("name", "")
        if name.startswith("models/"):
            name = name.split("/", 1)[1]
        if name:
            models.append(name)

    return models


def _is_text_gemini_model(name: str) -> bool:
    lowered = (name or "").lower()
    blocked_tokens = (
        "image",
        "tts",
        "computer-use",
        "robotics",
        "lyria",
        "gemma",
        "banana",
        "customtools",
    )
    return not any(token in lowered for token in blocked_tokens)


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

def safe_execute(code_string: str, keys=None):
    import codecs
    from io import StringIO
    from contextlib import redirect_stdout
    import matplotlib
    matplotlib.use('Agg')

    new_code_string = codecs.decode(code_string, 'unicode_escape')
    new_code_string = new_code_string.strip()

    if new_code_string.startswith("```python"):
        new_code_string = new_code_string[len("```python"):].strip()
    elif new_code_string.startswith("```"):
        new_code_string = new_code_string[len("```"):].strip()

    if new_code_string.endswith("```"):
        new_code_string = new_code_string[:-3].strip()

    output = None
    error_message = None

    try:
        buffer = StringIO()
        exec_globals = {"__name__": "__main__"}
        with redirect_stdout(buffer):
            exec(new_code_string, exec_globals)
        output = buffer.getvalue()
    except Exception as e:
        error_message = str(e)

    return output, error_message

def get_llama_response(tokenizer, pipeline, prompt, temperature=0.5):
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


def get_textdavinci002_response(prompt, temperature, max_tokens, n=1, patience=1, sleep_time=0):
    deployment = _get_env("OPENAI_TEXTDAVC002_DEPLOYMENT_NAME", required=True)
    last_error = None

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
    kwargs = {
        "model": AZURE_DEPLOYMENT_NAME,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
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

    raise RuntimeError(f"Azure chat code call failed after retries: {last_error}")


def get_gemini_response(full_prompt):
    if not GOOGLE_API_KEY:
        raise RuntimeError("GOOGLE_API_KEY is missing from .env")

    def candidate_model_names():
        seen = set()

        def add(name, bucket):
            normalized = _normalize_gemini_model_name(name)
            if normalized and normalized not in seen:
                seen.add(normalized)
                bucket.append(normalized)

        configured = []
        discovered = []
        fallbacks = []

        add(GEMINI_MODEL_NAME, configured)

        available_models = []
        try:
            available_models = [name for name in _list_gemini_models_via_rest() if _is_text_gemini_model(name)]
        except Exception:
            available_models = []

        configured_name = _normalize_gemini_model_name(GEMINI_MODEL_NAME)
        if configured_name and available_models:
            cfg_tokens = [token for token in re.split(r"[-_\s]+", configured_name) if token]
            for model_name in available_models:
                lowered = model_name.lower()
                if all(token in lowered for token in cfg_tokens if token not in {"gemini"}):
                    add(model_name, discovered)

        preferred_patterns = [
            "gemini-3.1-pro-preview",
            "gemini-3-flash-preview",
            "gemini-2.5-pro",
            "gemini-2.5-flash",
            "gemini-2.0-flash",
            "gemini-1.5-pro",
            "gemini-1.5-flash",
            "gemini-pro",
        ]
        for pattern in preferred_patterns:
            for model_name in available_models:
                if pattern in model_name:
                    add(model_name, discovered)

        for name in [
            "gemini-3.1-pro-preview",
            "gemini-3-flash-preview",
            "gemini-2.5-pro",
            "gemini-2.5-flash",
            "gemini-2.0-flash",
            "gemini-1.5-pro",
            "gemini-1.5-flash",
            "gemini-pro",
        ]:
            add(name, fallbacks)

        for bucket in (configured, discovered, fallbacks):
            for model_name in bucket:
                yield model_name

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

    def _azure_fallback():
        if not GEMINI_FALLBACK_TO_AZURE:
            return None
        messages = [{"role": "user", "content": full_prompt}]
        return _chat_completion(messages=messages, temperature=0, max_tokens=5000)

    errors = []

    available_models_for_error = []
    try:
        available_models_for_error = [name for name in _list_gemini_models_via_rest() if _is_text_gemini_model(name)]
    except Exception as exc:
        errors.append(f"list-models: {exc}")

    # Prefer the SDK when available.
    if hasattr(genai, "GenerativeModel"):
        for model_name in candidate_model_names():
            for attempt in range(3):
                try:
                    gemini_model = genai.GenerativeModel(model_name)
                    response = gemini_model.generate_content(full_prompt)
                    text = extract_text_from_response(response)
                    if text:
                        return text
                    errors.append(f"{model_name}: empty response")
                    break
                except Exception as exc:
                    message = str(exc)
                    retryable = "429" in message or "503" in message or "ResourceExhausted" in message
                    if retryable and attempt < 2:
                        time.sleep(2 * (attempt + 1))
                        continue
                    errors.append(f"{model_name}: {exc}")
                    break

    # Fallback to the REST API to avoid SDK-version incompatibilities.
    for model_name in candidate_model_names():
        for attempt in range(3):
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
                response = requests.post(url, json=payload, timeout=120)

                if response.status_code in (429, 503) and attempt < 2:
                    time.sleep(2 * (attempt + 1))
                    continue

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
                break
            except Exception as exc:
                message = str(exc)
                retryable = "429" in message or "503" in message
                if retryable and attempt < 2:
                    time.sleep(2 * (attempt + 1))
                    continue
                errors.append(f"{model_name}: {exc}")
                break

    if available_models_for_error:
        errors.append("available-models: " + ", ".join(sorted(available_models_for_error)))

    if GEMINI_FALLBACK_TO_AZURE:
        try:
            return _azure_fallback()
        except Exception as exc:
            errors.append(f"azure-fallback: {exc}")

    raise RuntimeError("Gemini call failed. Attempts: " + " | ".join(errors))


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
        response = requests.get(endpoint, headers=headers, params=params, timeout=30)
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
    response = requests.get(url, timeout=30)
    soup = BeautifulSoup(response.text, "html.parser")
    text = soup.get_text()
    lines = text.splitlines()
    logging.info("%s", lines[1000:])

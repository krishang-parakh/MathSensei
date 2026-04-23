import logging
import os
import re
import time
import unicodedata
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import requests
try:
    import xmltodict
except Exception:
    xmltodict = None

try:
    from wolframalpha import Document
except Exception:
    Document = None


WOLFRAM_QUERY_URL = "https://api.wolframalpha.com/v2/query"
WOLFRAM_RESULT_URL = "https://api.wolframalpha.com/v1/result"
_PROXY_ENV_NAMES = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
)

SHORT_ANSWER_FAILURE_MARKERS = (
    "did not understand your input",
    "no short answer available",
    "error ",
    "assumption",
    "computation timed out",
    "standard computation time exceeded",
)

BLOCKED_REQUEST_MARKERS = (
    "blocked request",
    "request blocked",
)

PREFERRED_POD_TITLES = (
    "result",
    "exact result",
    "decimal approximation",
    "solutions",
    "solution",
    "real solutions",
    "real solution",
    "values",
    "value",
    "answer",
    "simplification",
    "simplified form",
    "numerical result",
)

DEPRIORITIZED_POD_TITLES = (
    "input interpretation",
    "input",
    "notes",
    "definition",
)

_UNIT_AFTER_NUMBER_RE = re.compile(
    r"(?P<number>\b\d+(?:\.\d+)?(?:/\d+(?:\.\d+)?)?)"
    r"\s+"
    r"(?P<unit>[A-Za-z][A-Za-z0-9]*(?:/[A-Za-z][A-Za-z0-9]*)*"
    r"(?:\s+[A-Za-z][A-Za-z0-9]*(?:/[A-Za-z][A-Za-z0-9]*)*)*)"
    r"(?=\s*(?:[+\-*/^(),]|$))"
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


def _safe_response_text(response: Optional[requests.Response], limit: int = 240) -> str:
    if response is None:
        return ""
    try:
        text = (response.text or "").strip()
    except Exception:
        return ""
    if len(text) > limit:
        return text[:limit].rstrip() + "..."
    return text


def _looks_like_blocked_message(message: Optional[str]) -> bool:
    lowered = str(message or "").lower()
    return any(marker in lowered for marker in BLOCKED_REQUEST_MARKERS)


def _looks_like_blocked_http_error(status: Optional[int], body: Optional[str]) -> bool:
    if status != 403:
        return False
    return _looks_like_blocked_message(body) or not str(body or "").strip()


def _format_blocked_error(status: Optional[int], body: Optional[str]) -> str:
    code = f"HTTP {status}" if status is not None else "HTTP error"
    detail = str(body or "").strip() or "Blocked request"
    return (
        f"Wolfram request blocked ({code}): {detail}. "
        "Check WOLFRAM_ALPHA_APPID/account limits and network/IP policy."
    )


def _extract_v2_error_message(result: Any) -> Optional[str]:
    if not isinstance(result, dict):
        return None

    error_flag = str(result.get("@error") or "").strip().lower()
    raw_error = result.get("error")
    if error_flag not in {"true", "1", "yes"} and raw_error in (None, "", []):
        return None

    messages: List[str] = []

    def _collect(payload: Any) -> None:
        if isinstance(payload, dict):
            for key in ("msg", "@msg", "message", "@message", "code", "@code"):
                value = payload.get(key)
                if value not in (None, ""):
                    messages.append(str(value).strip())
            nested = payload.get("error")
            if nested is not None and nested is not payload:
                _collect(nested)
        elif isinstance(payload, list):
            for item in payload:
                _collect(item)
        elif payload not in (None, ""):
            messages.append(str(payload).strip())

    _collect(raw_error)
    for key in ("@datatypes", "@timedout"):
        value = result.get(key)
        if value not in (None, ""):
            messages.append(str(value).strip())

    merged = "; ".join(part for part in messages if part)
    return merged or "Wolfram v2 returned an error response"


def clean_wolfram_query(query: Optional[str]) -> Optional[str]:
    if query is None:
        return None

    q = unicodedata.normalize("NFKC", str(query)).strip()
    if not q:
        return None

    if q.startswith("```python"):
        q = q[len("```python"):].strip()
    elif q.startswith("```"):
        q = q[len("```"):].strip()
    if q.endswith("```"):
        q = q[:-3].strip()

    q = q.replace("`", "")
    # Preserve exponent intent from Python/SymPy-style "**" while still stripping markdown-ish noise.
    q = q.replace("**", "^")
    q = q.replace("$", "")
    q = q.replace("\\[", "").replace("\\]", "")
    q = q.replace("\u200b", "").replace("\ufeff", "")
    # Normalize common planner punctuation and units.
    q = q.replace("->", " ").replace("\u2192", " ")
    q = q.replace("\u00b0", "")

    replacements = {
        "\u2212": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u00d7": "*",
        "\u00f7": "/",
        "\u2264": "<=",
        "\u2265": ">=",
        "\u2260": "!=",
        "\u03c0": "pi",
        "\u2261": "<=>",
        "\u2283": "=>",
        "\u2022": " and ",
        "\u22c5": " and ",
        "\u2227": " and ",
        "\u2228": " or ",
        "\u223c": " not ",
        "\u00ac": " not ",
        "\u2200": "forall ",
        "\u2203": "exists ",
    }
    for old, new in replacements.items():
        q = q.replace(old, new)

    while q.startswith("*") or q.startswith("-"):
        q = q[1:].strip()

    if q.startswith("\\(") and q.endswith("\\)"):
        q = q[2:-2].strip()
    if q.startswith("$") and q.endswith("$"):
        q = q[1:-1].strip()
    if q.startswith("{") and q.endswith("}"):
        q = q[1:-1].strip()
    if (q.startswith('"') and q.endswith('"')) or (q.startswith("'") and q.endswith("'")):
        q = q[1:-1].strip()

    q = re.sub(r"(?<=\d),(?=\d)", "", q)

    while q and q[-1] in [".", ",", ";", ":"]:
        q = q[:-1].strip()

    q = " ".join(q.split())
    return q or None


def _build_unitless_arithmetic_candidate(query: Optional[str]) -> Optional[str]:
    cleaned = clean_wolfram_query(query)
    if not cleaned or not re.search(r"\d", cleaned) or not re.search(r"[A-Za-z]", cleaned):
        return None

    rewritten = _UNIT_AFTER_NUMBER_RE.sub(lambda match: match.group("number"), cleaned)
    rewritten = " ".join(rewritten.split())
    if rewritten == cleaned:
        return None
    return clean_wolfram_query(rewritten)


def _build_truth_table_candidate(query: Optional[str]) -> Optional[str]:
    cleaned = clean_wolfram_query(query)
    if not cleaned:
        return None

    lowered = cleaned.lower()
    idx = lowered.find("truth table")
    if idx < 0:
        return None

    tail = cleaned[idx + len("truth table") :].strip()
    if tail.lower().startswith("for "):
        tail = tail[4:].strip()
    tail = tail.strip(" :")
    if not tail:
        return "truth table"

    parts = [part.strip() for part in re.split(r"\s*,\s*", tail) if part.strip()]
    if len(parts) >= 2:
        return "truth table {" + ", ".join(parts) + "}"
    return f"truth table {tail}"


def _build_query_candidates(query: Optional[str]) -> List[str]:
    cleaned = clean_wolfram_query(query)
    if not cleaned:
        return []

    candidates: List[str] = []

    def add(candidate: Optional[str]) -> None:
        normalized = clean_wolfram_query(candidate)
        if normalized and normalized not in candidates:
            candidates.append(normalized)

    def _reorder_trailing_coefficient(text: Optional[str]) -> Optional[str]:
        if not text:
            return None
        lowered = text.lower()
        idx = lowered.rfind("coefficient of")
        if idx <= 0:
            return None
        expr = text[:idx].strip()
        coef = text[idx:].strip()
        if not expr or not coef:
            return None
        # Avoid double-inserting "in" if the coefficient phrase already includes it.
        if " in " in coef.lower():
            return None
        return f"{coef} in {expr}"

    add(cleaned)
    add(cleaned.replace(", ", ","))
    # Wolfram Alpha often fails on Python-style explicit multiplication ("7*x^4", "(...)*(...)").
    # Add a rewrite candidate that uses implicit multiplication ("7x^4", "(...)(...)").
    starless = re.sub(r"\s*\*\s*", " ", cleaned.replace("**", "^"))
    starless = re.sub(r"\)\s+\(", ")(", starless)
    starless = re.sub(r"(\d)\s+([A-Za-z])", r"\1\2", starless)
    starless = " ".join(starless.split())
    add(starless)
    add(_reorder_trailing_coefficient(cleaned))
    add(_reorder_trailing_coefficient(starless))
    add(_build_unitless_arithmetic_candidate(cleaned))
    add(_build_truth_table_candidate(cleaned))

    if cleaned.startswith("solve ") and " using Wolfram Alpha code" in cleaned:
        add(cleaned.replace(" using Wolfram Alpha code", ""))

    return candidates


def _query_wolfram_v2(
    session: requests.Session,
    app_id: str,
    query: str,
    timeout: int,
    retry_count: int = 0,
    max_retries: int = 2,
) -> Dict[str, Any]:
    """Query Wolfram Alpha v2 API with improved error handling."""
    if xmltodict is None or Document is None:
        raise RuntimeError("Wolfram Alpha parsing dependencies are not installed")
    
    try:
        response = session.get(
            WOLFRAM_QUERY_URL,
            # Prefer plaintext so downstream extraction doesn't depend on image-only pods.
            params={"appid": app_id, "input": query, "format": "plaintext"},
            timeout=timeout,
        )
        response.raise_for_status()
        doc = xmltodict.parse(response.text, postprocessor=Document.make)
        return doc["queryresult"]
    except requests.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else None
        # Retry on server errors (5xx) or rate limit errors (429)
        if status and (status == 429 or 500 <= status < 600) and retry_count < max_retries:
            raise exc  # Let caller handle retry
        # Don't retry on client errors (4xx except 429)
        raise exc
    except requests.Timeout:
        if retry_count < max_retries:
            raise
        raise RuntimeError(f"Wolfram Alpha query timed out after {max_retries} retries")
    except Exception as exc:
        raise RuntimeError(f"Failed to parse Wolfram Alpha response: {exc}")


def _query_wolfram_short_answer(
    session: requests.Session,
    app_id: str,
    query: str,
    timeout: int,
    retry_count: int = 0,
    max_retries: int = 2,
) -> Optional[str]:
    """Query Wolfram Alpha short answer API with improved error handling."""
    try:
        response = session.get(
            WOLFRAM_RESULT_URL,
            params={"appid": app_id, "i": query},
            timeout=timeout,
        )
        response.raise_for_status()
        answer = response.text.strip()
        if not answer:
            return None

        lowered = answer.lower()
        if any(marker in lowered for marker in SHORT_ANSWER_FAILURE_MARKERS):
            return None

        return answer
    except requests.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else None
        # Retry on server errors (5xx) or rate limit errors (429)
        if status and (status == 429 or 500 <= status < 600) and retry_count < max_retries:
            raise exc  # Let caller handle retry
        raise exc
    except requests.Timeout:
        if retry_count < max_retries:
            raise
        raise RuntimeError(f"Wolfram Alpha short answer query timed out after {max_retries} retries")


def _build_short_answer_result(query: str, answer: str) -> Dict[str, Any]:
    return {
        "@success": True,
        "@inputstring": query,
        "pod": [
            {
                "@title": "Result",
                "@primary": True,
                "subpod": {"plaintext": answer},
            }
        ],
    }


def extract_wolfram_plaintext_answer(result: Any) -> Optional[str]:
    """Extract plaintext answer from Wolfram Alpha result with enhanced parsing."""
    if not isinstance(result, dict):
        return None

    pods = result.get("pod") or []
    if isinstance(pods, dict):
        pods = [pods]

    ranked_candidates = []
    fallback_candidates = []

    for pod in pods:
        if not isinstance(pod, dict):
            continue

        title = str(pod.get("@title") or pod.get("title") or "").strip().lower()
        primary = str(pod.get("@primary") or pod.get("primary") or "").strip().lower() in {"true", "1", "yes"}
        subpods = pod.get("subpod") or []
        if isinstance(subpods, dict):
            subpods = [subpods]

        for subpod in subpods:
            if not isinstance(subpod, dict):
                continue
            plaintext = subpod.get("plaintext")
            if plaintext in (None, ""):
                continue
            answer = str(plaintext).strip()
            if not answer:
                continue
            
            # Filter out empty responses or error messages
            if len(answer) < 2 or answer.lower() in ("none", "n/a", "undefined"):
                continue
            
            # Skip if plaintext contains only whitespace or special characters
            if not any(c.isalnum() for c in answer):
                continue

            if any(token in title for token in DEPRIORITIZED_POD_TITLES):
                fallback_candidates.append(answer)
                continue

            score = 2
            if primary:
                score = 0
            elif any(token in title for token in PREFERRED_POD_TITLES):
                score = 1
            ranked_candidates.append((score, answer))

    if ranked_candidates:
        ranked_candidates.sort(key=lambda item: item[0])
        return ranked_candidates[0][1]

    if fallback_candidates:
        return fallback_candidates[0]

    return None


def query_wolfram_alpha(
    app_id: str,
    query: Optional[str],
    logger: Optional[logging.Logger] = None,
    timeout: int = 20,
    max_attempts: int = 3,
    initial_delay: float = 0.5,
) -> Dict[str, Any]:
    """
    Query Wolfram Alpha with improved retry logic and rate limiting.
    
    Args:
        app_id: Wolfram Alpha API key
        query: Query string
        logger: Optional logger instance
        timeout: Request timeout in seconds
        max_attempts: Maximum number of attempts per candidate query
        initial_delay: Initial delay in seconds for exponential backoff
    
    Returns:
        Dictionary with query, result, answer, source, and error fields
    """
    log = logger or logging.getLogger(__name__)
    candidates = _build_query_candidates(query)

    if not candidates:
        return {
            "query": None,
            "result": None,
            "answer": None,
            "source": None,
            "error": "Empty Wolfram Alpha query",
        }

    if xmltodict is None or Document is None:
        return {
            "query": candidates[0],
            "result": None,
            "answer": None,
            "source": None,
            "error": "Wolfram Alpha dependencies are not installed",
        }

    session = requests.Session()
    session.trust_env = not _should_bypass_env_proxies()
    last_error: Optional[str] = None
    
    # Configure session with retry strategy
    session.headers.update({
        "User-Agent": "MathSensei/1.0",
    })

    try:
        for candidate in candidates:
            v2_unusable = False
            for attempt in range(1, max_attempts + 1):
                try:
                    result = _query_wolfram_v2(session, app_id, candidate, timeout)
                    v2_error_message = _extract_v2_error_message(result)
                    if v2_error_message:
                        if _looks_like_blocked_message(v2_error_message):
                            last_error = _format_blocked_error(403, v2_error_message)
                            log.error("Wolfram Alpha request blocked for query: %s", candidate[:200])
                            return {
                                "query": candidate,
                                "result": None,
                                "answer": None,
                                "source": None,
                                "error": last_error,
                            }
                        last_error = f"Wolfram v2 error: {v2_error_message}"
                        v2_unusable = True
                        break

                    extracted = extract_wolfram_plaintext_answer(result)
                    if extracted:
                        return {
                            "query": candidate,
                            "result": result,
                            "answer": extracted,
                            "source": "v2",
                            "error": None,
                        }

                    # v2 returned a well-formed response but no plaintext pods. Try the short-answer API,
                    # but if it fails (e.g. unsupported query -> 501) keep the v2 result so downstream
                    # can still attempt an LLM-based extraction.
                    try:
                        short_answer = _query_wolfram_short_answer(session, app_id, candidate, timeout)
                    except Exception:
                        short_answer = None

                    if short_answer:
                        log.info("Wolfram Alpha short-answer fallback succeeded for query: %s", candidate[:200])
                        return {
                            "query": candidate,
                            "result": _build_short_answer_result(candidate, short_answer),
                            "answer": short_answer,
                            "source": "v1-result",
                            "error": None,
                        }

                    pods = result.get("pod") if isinstance(result, dict) else None
                    if not pods:
                        # v2 parsed but produced no pods (common when WA can't interpret Python-ish syntax).
                        # Try the next candidate rewrite instead of returning a blank result.
                        last_error = "Wolfram v2 returned no pods"
                        v2_unusable = True
                        break

                    return {
                        "query": candidate,
                        "result": result,
                        "answer": None,
                        "source": "v2",
                        "error": None,
                    }
                except requests.HTTPError as exc:
                    status = exc.response.status_code if exc.response is not None else None
                    body = _safe_response_text(exc.response)
                    if _looks_like_blocked_http_error(status, body):
                        last_error = _format_blocked_error(status, body)
                        log.error("Wolfram Alpha request blocked for query: %s", candidate[:200])
                        return {
                            "query": candidate,
                            "result": None,
                            "answer": None,
                            "source": None,
                            "error": last_error,
                        }
                    
                    if status == 429:  # Rate limit
                        # Extract retry-after header if available
                        retry_after = exc.response.headers.get("Retry-After")
                        delay = float(retry_after) if retry_after else min(initial_delay * (2 ** (attempt - 1)), 30)
                        last_error = f"Rate limited (HTTP 429): {exc}"
                        if attempt < max_attempts:
                            log.warning(
                                "Wolfram Alpha rate limited. Retrying after %s seconds (attempt %s/%s)",
                                delay,
                                attempt,
                                max_attempts,
                            )
                            time.sleep(delay)
                            continue
                        break
                    
                    elif status and 500 <= status < 600:  # Server error
                        delay = min(initial_delay * (2 ** (attempt - 1)), 10)
                        last_error = f"HTTP {status}: {exc}"
                        if attempt < max_attempts:
                            log.warning(
                                "Wolfram Alpha v2 server error (HTTP %s). Retrying in %s seconds (attempt %s/%s)",
                                status,
                                delay,
                                attempt,
                                max_attempts,
                            )
                            time.sleep(delay)
                            continue
                        break
                    
                    else:  # Client error (4xx except 429)
                        if body:
                            last_error = f"Wolfram v2 API failed: HTTP {status}: {body}"
                        else:
                            last_error = f"Wolfram v2 API failed: HTTP {status}"
                        log.error("Wolfram Alpha v2 client error (HTTP %s) for query: %s", status, candidate[:200])
                        break
                
                except requests.Timeout as exc:
                    last_error = f"Timeout: {exc}"
                    if attempt < max_attempts:
                        delay = min(initial_delay * 2, 5)
                        log.warning(
                            "Wolfram Alpha v2 timeout. Retrying in %s seconds (attempt %s/%s)",
                            delay,
                            attempt,
                            max_attempts,
                        )
                        time.sleep(delay)
                        continue
                    break
                
                except requests.RequestException as exc:
                    last_error = str(exc)
                    if attempt < max_attempts:
                        delay = min(initial_delay * (2 ** (attempt - 1)), 5)
                        log.warning(
                            "Wolfram Alpha v2 request error. Retrying in %s seconds (attempt %s/%s)",
                            delay,
                            attempt,
                            max_attempts,
                        )
                        time.sleep(delay)
                        continue
                    break

            if v2_unusable:
                continue

            # Fallback to short answer API
            try:
                answer = _query_wolfram_short_answer(session, app_id, candidate, timeout)
            except requests.HTTPError as exc:
                status = exc.response.status_code if exc.response is not None else None
                body = _safe_response_text(exc.response)
                if _looks_like_blocked_http_error(status, body):
                    last_error = _format_blocked_error(status, body)
                    return {
                        "query": candidate,
                        "result": None,
                        "answer": None,
                        "source": None,
                        "error": last_error,
                    }
                if status == 429:
                    retry_after = exc.response.headers.get("Retry-After")
                    delay = float(retry_after) if retry_after else 2
                    log.warning("Wolfram Alpha short answer API rate limited. Waiting %s seconds.", delay)
                    time.sleep(delay)
                    try:
                        answer = _query_wolfram_short_answer(session, app_id, candidate, timeout)
                    except Exception:
                        answer = None
                else:
                    if body:
                        last_error = f"Wolfram short-answer API failed: HTTP {status}: {body}"
                    else:
                        last_error = f"Wolfram short-answer API failed: HTTP {status}"
                    answer = None
            except requests.RequestException as exc:
                last_error = str(exc)
                answer = None

            if answer:
                log.info("Wolfram Alpha short-answer fallback succeeded for query: %s", candidate[:200])
                return {
                    "query": candidate,
                    "result": _build_short_answer_result(candidate, answer),
                    "answer": answer,
                    "source": "v1-result",
                    "error": None,
                }
    finally:
        session.close()

    return {
        "query": candidates[0] if candidates else None,
        "result": None,
        "answer": None,
        "source": None,
        "error": last_error or "Unknown Wolfram Alpha error",
    }

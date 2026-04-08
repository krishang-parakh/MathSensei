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
    q = q.replace("**", "")
    q = q.replace("$", "")
    q = q.replace("\\[", "").replace("\\]", "")
    q = q.replace("\u200b", "").replace("\ufeff", "")

    replacements = {
        "−": "-",
        "–": "-",
        "—": "-",
        "×": "*",
        "÷": "/",
        "≤": "<=",
        "≥": ">=",
        "≠": "!=",
        "π": "pi",
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


def _build_query_candidates(query: Optional[str]) -> List[str]:
    cleaned = clean_wolfram_query(query)
    if not cleaned:
        return []

    candidates: List[str] = []

    def add(candidate: Optional[str]) -> None:
        normalized = clean_wolfram_query(candidate)
        if normalized and normalized not in candidates:
            candidates.append(normalized)

    add(cleaned)
    add(cleaned.replace(", ", ","))
    add(_build_unitless_arithmetic_candidate(cleaned))

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
            params={"appid": app_id, "input": query},
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
            for attempt in range(1, max_attempts + 1):
                try:
                    result = _query_wolfram_v2(session, app_id, candidate, timeout)
                    return {
                        "query": candidate,
                        "result": result,
                        "answer": None,
                        "source": "v2",
                        "error": None,
                    }
                except requests.HTTPError as exc:
                    status = exc.response.status_code if exc.response is not None else None
                    
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
                        last_error = f"HTTP {status}: {exc}"
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

            # Fallback to short answer API
            try:
                answer = _query_wolfram_short_answer(session, app_id, candidate, timeout)
            except requests.HTTPError as exc:
                status = exc.response.status_code if exc.response is not None else None
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
                    last_error = f"Short answer API failed: HTTP {status}"
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

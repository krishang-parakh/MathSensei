import logging
import re
import time
import unicodedata
from typing import Any, Dict, List, Optional

import requests
import xmltodict
from wolframalpha import Document


WOLFRAM_QUERY_URL = "https://api.wolframalpha.com/v2/query"
WOLFRAM_RESULT_URL = "https://api.wolframalpha.com/v1/result"

SHORT_ANSWER_FAILURE_MARKERS = (
    "did not understand your input",
    "no short answer available",
    "error ",
)


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

    if cleaned.startswith("solve ") and " using Wolfram Alpha code" in cleaned:
        add(cleaned.replace(" using Wolfram Alpha code", ""))

    return candidates


def _query_wolfram_v2(session: requests.Session, app_id: str, query: str, timeout: int) -> Dict[str, Any]:
    response = session.get(
        WOLFRAM_QUERY_URL,
        params={"appid": app_id, "input": query},
        timeout=timeout,
    )
    response.raise_for_status()
    doc = xmltodict.parse(response.text, postprocessor=Document.make)
    return doc["queryresult"]


def _query_wolfram_short_answer(
    session: requests.Session, app_id: str, query: str, timeout: int
) -> Optional[str]:
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


def query_wolfram_alpha(
    app_id: str,
    query: Optional[str],
    logger: Optional[logging.Logger] = None,
    timeout: int = 20,
    max_attempts: int = 3,
) -> Dict[str, Any]:
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

    session = requests.Session()
    last_error: Optional[str] = None

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
                    last_error = f"HTTP {status}: {exc}"
                    if status and 500 <= status < 600 and attempt < max_attempts:
                        log.warning(
                            "Wolfram Alpha v2 attempt %s/%s failed with HTTP %s for query: %s",
                            attempt,
                            max_attempts,
                            status,
                            candidate[:200],
                        )
                        time.sleep(min(1.0 * attempt, 3.0))
                        continue
                    break
                except requests.RequestException as exc:
                    last_error = str(exc)
                    if attempt < max_attempts:
                        time.sleep(min(1.0 * attempt, 3.0))
                        continue
                    break

            try:
                answer = _query_wolfram_short_answer(session, app_id, candidate, timeout)
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
        "query": candidates[0],
        "result": None,
        "answer": None,
        "source": None,
        "error": last_error or "Unknown Wolfram Alpha error",
    }

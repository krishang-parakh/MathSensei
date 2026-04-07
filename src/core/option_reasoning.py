import re
import unicodedata
from typing import Iterable, List, Optional

from core.answer_parsing import (
    clean_answer_text,
    extract_answer_candidates,
    extract_final_answer_option_letter,
    extract_numeric_answer,
    extract_option_letter,
)

try:
    from sympy import N
    from sympy.parsing.sympy_parser import (
        convert_xor,
        implicit_multiplication_application,
        parse_expr,
        standard_transformations,
    )
except Exception:  # pragma: no cover - exercised when SymPy is unavailable
    N = None
    parse_expr = None
    standard_transformations = ()
    implicit_multiplication_application = None
    convert_xor = None


_OPTION_LETTER_RE = re.compile(r"^\s*(?:option\s+)?([A-Z])[\)\.\:\s-]*(.*)$", re.IGNORECASE)
_OPTION_LISTING_RE = re.compile(r"^\s*(?:option\s+)?[A-Z][\)\.\:]", re.IGNORECASE)
_RESOLVED_OPTION_PREFIX_RE = re.compile(
    r"^\s*(?:resolved option by numeric comparison|resolved option|numeric option cross-check)\s*:\s*",
    re.IGNORECASE,
)
_TRAILING_UNITS_RE = re.compile(r"\s+(?:[A-Za-z$][A-Za-z/%$,\.\-\s]{0,40})$")
_PERCENT_RE = re.compile(r"(?<=\d)\s*%")


def option_entries(options) -> List[dict]:
    entries = []
    for idx, option in enumerate(options or []):
        if isinstance(option, dict):
            key = clean_answer_text(option.get("key") or option.get("option") or option.get("id") or chr(ord("A") + idx))
            label = clean_answer_text(option.get("label") or option.get("text") or option.get("value") or "")
        else:
            option_text = clean_answer_text(option)
            match = _OPTION_LETTER_RE.match(option_text or "")
            if match:
                key = match.group(1).upper()
                label = clean_answer_text(match.group(2))
            else:
                key = chr(ord("A") + idx)
                label = option_text
        if key and label:
            entries.append({"key": str(key).strip().upper(), "label": str(label).strip()})
    return entries


def question_requests_closest_option(question_text: Optional[str]) -> bool:
    cleaned = clean_answer_text(question_text)
    if not cleaned:
        return False

    lowered = cleaned.lower()
    phrases = (
        "closest estimate",
        "best estimate",
        "closest approximation",
        "best approximation",
        "approximately how",
        "approximately what",
        "approximate value",
        "approximate answer",
        "closest to",
        "closest expression",
        "closest value",
        "nearest estimate",
        "nearest value",
        "which estimate",
    )
    if any(phrase in lowered for phrase in phrases):
        return True

    keyword_count = sum(keyword in lowered for keyword in ("closest", "estimate", "approximate", "approximately", "nearest"))
    return keyword_count >= 2


def program_uses_unsupported_closest_estimate_rounding(question_text: Optional[str], program_text: Optional[str]) -> bool:
    if not question_requests_closest_option(question_text):
        return False

    question = clean_answer_text(question_text)
    program = clean_answer_text(program_text)
    if not question or not program:
        return False

    lowered_question = question.lower()
    explicit_rounding_instruction = any(
        phrase in lowered_question
        for phrase in (
            "round to",
            "rounded to",
            "round each",
            "nearest ten",
            "nearest hundred",
            "nearest thousand",
            "nearest whole",
            "nearest integer",
            "nearest tenth",
            "nearest hundredth",
            "nearest thousandth",
        )
    )
    if explicit_rounding_instruction:
        return False

    lowered_program = program.lower()
    return any(token in lowered_program for token in ("round(", ".round(", "floor(", "ceil(", "quantize("))


def extract_numeric_expression_value(text: Optional[str]) -> Optional[float]:
    cleaned = clean_answer_text(text)
    if not cleaned:
        return None

    candidate = unicodedata.normalize("NFKC", cleaned)
    candidate = re.sub(r"^\s*(?:final answer|the correct answer is|the answer is|answer)\s*:\s*", "", candidate, flags=re.IGNORECASE)
    candidate = re.sub(r"^\s*(?:therefore|thus|hence|so)\b[^:\n]{0,40}:\s*", "", candidate, flags=re.IGNORECASE)
    candidate = _RESOLVED_OPTION_PREFIX_RE.sub("", candidate)
    candidate = candidate.strip().strip("`").strip("*_").strip()
    if not candidate:
        return None

    lowered = candidate.lower()
    if lowered in {"none of these", "none"}:
        return None

    candidate = candidate.replace("−", "-").replace("–", "-").replace("—", "-")
    candidate = candidate.replace("×", "*").replace("÷", "/")
    candidate = candidate.replace("√", "sqrt")
    candidate = candidate.replace("$", "")
    candidate = candidate.replace(",", "")
    candidate = _PERCENT_RE.sub("", candidate)
    candidate = _TRAILING_UNITS_RE.sub("", candidate).strip()

    if parse_expr is not None and candidate:
        try:
            transformations = standard_transformations + (implicit_multiplication_application, convert_xor)
            parsed = parse_expr(candidate, transformations=transformations, evaluate=True)
            return float(N(parsed, 15))
        except Exception:
            pass

    numeric = extract_numeric_answer(candidate)
    if numeric is None:
        return None
    try:
        return float(str(numeric).replace(",", ""))
    except (TypeError, ValueError):
        return None


def _normalize_option_label(text: Optional[str]) -> Optional[str]:
    cleaned = clean_answer_text(text)
    if not cleaned:
        return None

    value = unicodedata.normalize("NFKC", cleaned)
    value = re.sub(r"^\s*(?:final answer|the correct answer is|the answer is|answer)\s*:\s*", "", value, flags=re.IGNORECASE)
    value = _RESOLVED_OPTION_PREFIX_RE.sub("", value)
    value = re.sub(r"^\s*option\s+[A-Z][\)\.\:\s-]*", "", value, flags=re.IGNORECASE)
    value = re.sub(r"^\s*[A-Z][\)\.\:\s-]+", "", value)
    value = value.replace("`", "").replace("*", "").replace("_", "")
    value = re.sub(r"\s+", "", value.lower())
    return value.strip(".") or None


def _numeric_values_equivalent(left: float, right: float) -> bool:
    tolerance = 1e-9 * max(1.0, abs(left), abs(right))
    return abs(left - right) <= tolerance


def _exact_option_match(candidate: Optional[str], entries: Iterable[dict]) -> Optional[dict]:
    cleaned = clean_answer_text(candidate)
    if not cleaned:
        return None

    explicit = extract_final_answer_option_letter(cleaned) or extract_option_letter(cleaned)
    if explicit:
        for entry in entries:
            if entry["key"] == explicit:
                return {**entry, "match_type": "explicit-option", "candidate": cleaned}

    normalized_candidate = _normalize_option_label(cleaned)
    numeric_candidate = extract_numeric_expression_value(cleaned)
    if normalized_candidate is None and numeric_candidate is None:
        return None

    for entry in entries:
        label = entry["label"]
        if normalized_candidate is not None and normalized_candidate == _normalize_option_label(label):
            return {**entry, "match_type": "exact-text", "candidate": cleaned}

        numeric_label = extract_numeric_expression_value(label)
        if numeric_candidate is not None and numeric_label is not None and _numeric_values_equivalent(numeric_candidate, numeric_label):
            return {**entry, "match_type": "exact-numeric", "candidate": cleaned}

    return None


def _candidate_texts_for_resolution(text_values: Iterable[Optional[str]]) -> List[str]:
    candidates: List[str] = []

    def add(candidate: Optional[str]) -> None:
        cleaned = clean_answer_text(candidate)
        if cleaned and cleaned not in candidates:
            candidates.append(cleaned)

    for text in text_values:
        cleaned = clean_answer_text(text)
        if not cleaned:
            continue
        add(extract_final_answer_option_letter(cleaned))
        add(extract_option_letter(cleaned))
        for candidate in extract_answer_candidates(cleaned):
            add(candidate)
        add(cleaned)

    return candidates


def _looks_like_option_listing(candidate: Optional[str]) -> bool:
    cleaned = clean_answer_text(candidate)
    if not cleaned:
        return False
    if not _OPTION_LISTING_RE.match(cleaned):
        return False
    lowered = cleaned.lower()
    return "answer" not in lowered and "closest" not in lowered and "correct" not in lowered


def _is_bare_option_reference(candidate: Optional[str]) -> bool:
    cleaned = clean_answer_text(candidate)
    if not cleaned:
        return False

    stripped = cleaned.replace("`", "").replace("*", "").replace("_", "").strip().rstrip(".")
    if not stripped:
        return False

    patterns = (
        r"^(?:option\s+)?[A-Z]$",
        r"^(?:final answer|the correct answer is|the answer is|answer)\s*:?\s*[A-Z]$",
    )
    return any(re.fullmatch(pattern, stripped, flags=re.IGNORECASE) for pattern in patterns)


def _closest_numeric_option_match(candidates: Iterable[Optional[str]], numeric_options):
    for candidate in candidates:
        if _looks_like_option_listing(candidate):
            continue
        numeric_candidate = extract_numeric_expression_value(candidate)
        if numeric_candidate is None:
            continue

        ranked = sorted(
            (
                abs(numeric_candidate - numeric_option_value),
                entry["key"],
                entry,
                numeric_option_value,
            )
            for entry, numeric_option_value in numeric_options
        )
        _, _, best_entry, best_value = ranked[0]
        return {
            **best_entry,
            "match_type": "closest-numeric",
            "candidate": candidate,
            "candidate_value": numeric_candidate,
            "option_value": best_value,
            "delta": abs(numeric_candidate - best_value),
        }

    return None


def select_option_from_values(
    text_values: Iterable[Optional[str]],
    options,
    *,
    question_text: Optional[str] = None,
    allow_nearest: bool = True,
):
    """
    Resolve an answer text to an option key/label through a natural, linear process:
    1. Try exact matching on all candidates (most reliable)
    2. If no match and closest-numeric is allowed, find the numerically closest option
    3. Return the first matching result, or None if no match
    
    This design is intentionally simple to avoid conditional skips and multiple passes.
    """
    entries = option_entries(options)
    if not entries:
        return None

    candidates = _candidate_texts_for_resolution(text_values)
    
    # STEP 1: Try exact matching on all candidates (most reliable method)
    for candidate in candidates:
        matched = _exact_option_match(candidate, entries)
        if matched:
            return matched
    
    # STEP 2: Try closest-numeric matching (second-best for approximate answers)
    # Only attempt if: (a) caller allows it, (b) question requests it, (c) we have multiple options
    if allow_nearest and question_requests_closest_option(question_text):
        numeric_options = []
        for entry in entries:
            numeric_value = extract_numeric_expression_value(entry["label"])
            if numeric_value is not None and "none of these" not in entry["label"].lower():
                numeric_options.append((entry, numeric_value))
        
        # Only use closest-numeric if we have at least 2 numeric options to choose between
        if len(numeric_options) >= 2:
            closest_match = _closest_numeric_option_match(candidates, numeric_options)
            if closest_match:
                return closest_match
    
    return None

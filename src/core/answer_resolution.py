import re
from decimal import Decimal, InvalidOperation

from core.answer_parsing import (
    candidate_has_rejection_cue,
    candidate_looks_like_option_listing,
    clean_answer_text,
    extract_answer_candidates,
    extract_numeric_answer,
    extract_final_answer_option_letter,
    extract_option_letter,
    extract_preferred_answer,
    text_has_option_listing_block,
)
from core.option_reasoning import option_entries, select_option_from_values


OPTION_DATASETS = {"AQUA", "MMLU"}


def uses_option_answers(record):
    if isinstance(record, dict):
        dataset = record.get("dataset")
        options = record.get("options")
    else:
        dataset = record
        options = None
    return bool(options) or str(dataset or "").upper() in OPTION_DATASETS


def option_entry_for_key(options, option_key):
    key = clean_answer_text(option_key)
    if not key:
        return None
    normalized_key = key.upper()
    for entry in option_entries(options):
        if entry["key"] == normalized_key:
            return entry
    return None


def format_option_answer(options, option_key):
    entry = option_entry_for_key(options, option_key)
    if not entry:
        key = clean_answer_text(option_key)
        return key.upper() if key else None
    return f"{entry['key']}. {entry['label']}"


def _answer_bundle(value=None, option_entry=None):
    if option_entry:
        return {
            "value": option_entry["label"],
            "display": f"{option_entry['key']}. {option_entry['label']}",
            "option_key": option_entry["key"],
            "option_label": option_entry["label"],
        }

    cleaned = clean_answer_text(value)
    return {
        "value": cleaned,
        "display": cleaned,
        "option_key": None,
        "option_label": None,
    }


def _clean_texts(*values):
    cleaned = []
    for value in values:
        text = clean_answer_text(value)
        if text and text not in cleaned:
            cleaned.append(text)
    return cleaned


def _allowed_option_letters(options):
    letters = "".join(entry["key"] for entry in option_entries(options))
    return letters or "ABCDE"


def _is_bare_option_reference(text, options):
    cleaned = clean_answer_text(text)
    if not cleaned:
        return False

    stripped = cleaned.replace("**", "").replace("__", "").replace("`", "").strip().rstrip(".")
    if not stripped:
        return False

    allowed = _allowed_option_letters(options)
    explicit = extract_option_letter(stripped, allowed=allowed)
    if explicit and explicit == stripped.upper():
        return True

    patterns = (
        rf"^(?:option\s+)?[{allowed}]$",
        rf"^(?:final answer|the correct answer is|the answer is|answer)\s*:?\s*[{allowed}]$",
    )
    return any(re.fullmatch(pattern, stripped, flags=re.IGNORECASE) for pattern in patterns)


def _preferred_explicit_text(text):
    cleaned = clean_answer_text(text)
    if not cleaned:
        return None
    return clean_answer_text(extract_preferred_answer(cleaned)) or cleaned


def _preferred_solution_text(text):
    cleaned = clean_answer_text(text)
    if not cleaned:
        return None

    preferred = clean_answer_text(extract_preferred_answer(cleaned))
    if preferred:
        return preferred

    lines = [line.strip("- ").strip() for line in cleaned.splitlines() if line.strip()]
    for line in reversed(lines):
        lowered = line.lower()
        if "final answer" in lowered or "therefore" in lowered:
            return clean_answer_text(line)
    return clean_answer_text(lines[-1]) if lines else None


def _resolve_option_entry(priority_texts, fallback_texts, options, question_text=None):
    """
    Resolve texts to an option entry using a natural, linear process:
    1. Try to find explicit option letters in priority texts first
    2. Then try full resolution on priority + fallback texts
    3. If explicit option text conflicts with fallback-derived values, prefer the fallback-derived option
    """
    if not options:
        return None

    allowed = _allowed_option_letters(options)

    explicit_entry = None
    explicit_from_final_tag = False
    for text in _clean_texts(*priority_texts):
        explicit = extract_final_answer_option_letter(text, allowed=allowed)
        if explicit:
            explicit_from_final_tag = True
        cleaned_text = clean_answer_text(text)
        if not explicit and cleaned_text:
            # Multi-line solutions often contain a bare option reference on the last line
            # ("The answer is E.") which we still want to detect as an explicit letter.
            for candidate in extract_answer_candidates(cleaned_text):
                if not candidate:
                    continue
                if not _is_bare_option_reference(candidate, options):
                    continue
                explicit = extract_option_letter(candidate, allowed=allowed)
                if explicit:
                    if "final answer" in candidate.lower() or candidate.lstrip().startswith("####"):
                        explicit_from_final_tag = True
                    break

        if not explicit and cleaned_text and len(cleaned_text) <= 80 and "\n" not in cleaned_text:
            explicit = extract_option_letter(cleaned_text, allowed=allowed)
        entry = option_entry_for_key(options, explicit)
        if entry:
            explicit_entry = entry
            break

    combined_texts = _clean_texts(*priority_texts, *fallback_texts)
    matched = select_option_from_values(
        combined_texts,
        options,
        question_text=question_text,
        allow_nearest=True,
    )

    if explicit_entry:
        # If an explicit option letter is present, it can be wrong (e.g. "The answer is E")
        # even when the surrounding text derives a value matching a different option.
        # Prefer value-derived matches from the text (and/or tool outputs) over a conflicting
        # bare option reference.
        derived_candidates = []
        explicit_prefix = re.compile(rf"(?im)^\s*(?:option\s+)?[{re.escape(allowed)}][\)\.\:]\s+")
        for text in combined_texts:
            for candidate in extract_answer_candidates(text):
                if not candidate:
                    continue
                if candidate_has_rejection_cue(candidate):
                    continue
                if candidate_looks_like_option_listing(candidate, allowed=allowed):
                    continue
                if _is_bare_option_reference(candidate, options):
                    continue
                if explicit_prefix.match(candidate):
                    # Skip "E. 5,000" style explicit option labels; we already captured those.
                    continue
                derived_candidates.append(candidate)

        derived_match = select_option_from_values(
            derived_candidates,
            options,
            question_text=question_text,
            allow_nearest=True,
        )

        fallback_match = select_option_from_values(
            _clean_texts(*fallback_texts),
            options,
            question_text=question_text,
            allow_nearest=True,
        )
        # If the solution text included a real "Final Answer: <letter>" tag, don't let tool outputs
        # override it. Tool outputs often print per-option diagnostics (e.g. "A = 750", "B = 800")
        # that can spuriously match an option even when the solver finalized a different one.
        if (
            not explicit_from_final_tag
            and fallback_match
            and fallback_match.get("key") != explicit_entry.get("key")
        ):
            return fallback_match
        if (
            not explicit_from_final_tag
            and derived_match
            and derived_match.get("key") != explicit_entry.get("key")
        ):
            return derived_match
        return explicit_entry

    return matched


def resolve_final_answer_bundle(record):
    explicit = record.get("final_answer") or record.get("answer")
    solution = (
        record.get("final_generated_solution")
        or record.get("solution")
        or record.get("solution_generator_output")
    )
    options = record.get("options")
    question_text = record.get("problem") or record.get("question")
    fallback_texts = _clean_texts(
        record.get("program_output") or record.get("program_executor:output"),
        record.get("wolfram_output") or record.get("wolfram_alpha_search:output"),
    )

    if uses_option_answers(record):
        entry = _resolve_option_entry(
            [solution, explicit],
            fallback_texts,
            options,
            question_text=question_text,
        )
        if entry:
            return _answer_bundle(option_entry=entry)

    value = (
        _preferred_explicit_text(explicit)
        or _preferred_solution_text(solution)
        or next(
            (
                candidate
                for candidate in (_preferred_solution_text(text) for text in fallback_texts)
                if candidate
            ),
            None,
        )
    )

    # For non-option datasets, prefer tool-derived numerics when they strongly disagree with a
    # "pure numeric" solution snippet. This prevents bad SG extractions like "Answer: 0"
    # from overriding a correct Python/WA computation.
    dataset = str(record.get("dataset") or "").upper()

    def normalize_decimal(candidate):
        numeric = extract_numeric_answer(candidate or "")
        if not numeric:
            return None
        token = str(numeric).strip().replace(",", "")
        try:
            return Decimal(token)
        except (InvalidOperation, ValueError):
            return None

    def looks_like_pure_number(text):
        cleaned = clean_answer_text(text) or ""
        cleaned = cleaned.replace("\\boxed{", "").replace("}", "").strip()
        return bool(re.fullmatch(r"[-+]?\d[\d,]*(?:\.\d+)?", cleaned))

    if dataset == "MATH" and not uses_option_answers(record) and looks_like_pure_number(value):
        sol_dec = normalize_decimal(value)
        py_dec = normalize_decimal(record.get("program_output") or record.get("program_executor:output"))
        wa_dec = normalize_decimal(record.get("wolfram_output") or record.get("wolfram_alpha_search:output"))

        if py_dec is not None and wa_dec is not None and py_dec == wa_dec and sol_dec is not None and sol_dec != py_dec:
            return _answer_bundle(value=str(py_dec))
        if py_dec is not None and sol_dec is not None and sol_dec != py_dec and (wa_dec is None or wa_dec == py_dec):
            return _answer_bundle(value=str(py_dec))
        if wa_dec is not None and sol_dec is not None and sol_dec != wa_dec and py_dec is None:
            return _answer_bundle(value=str(wa_dec))

    return _answer_bundle(value=value)


def resolve_gold_answer_bundle(record):
    options = record.get("options")
    correct_option = record.get("correct_option")
    explicit = record.get("gold_answer")
    solution = record.get("ground_truth_solution")

    if uses_option_answers(record):
        entry = option_entry_for_key(options, correct_option) or _resolve_option_entry(
            [explicit, solution],
            [],
            options,
        )
        if entry:
            return _answer_bundle(option_entry=entry)

    value = _preferred_explicit_text(explicit) or _preferred_solution_text(solution)
    return _answer_bundle(value=value)

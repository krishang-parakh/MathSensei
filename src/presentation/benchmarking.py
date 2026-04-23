import csv
import json
import math
import os
import re
from datetime import datetime
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP

from core.answer_parsing import (
    extract_answer_candidates,
    extract_numeric_answer,
    extract_option_letter,
    number_words_to_numeric_string,
)
from core.answer_resolution import resolve_final_answer_bundle, resolve_gold_answer_bundle

try:
    from sympy import N, simplify
    from sympy.parsing.sympy_parser import (
        convert_xor,
        implicit_multiplication_application,
        parse_expr,
        standard_transformations,
    )
except Exception:
    N = None
    simplify = None
    parse_expr = None
    standard_transformations = ()
    implicit_multiplication_application = None
    convert_xor = None


def clean_benchmark_text(value):
    if value is None:
        return None
    return str(value).strip()


def dataset_label(dataset):
    if dataset == "ALL":
        return "MIXED"
    if not dataset:
        return "Unknown"
    return str(dataset).upper()


def _extract_option_letter(text):
    return extract_option_letter(text)


def _extract_gsm_number(text):
    return extract_numeric_answer(text)


def _candidate_texts(*values):
    candidates = []
    for value in values:
        cleaned = clean_benchmark_text(value)
        if not cleaned:
            continue
        extracted = extract_answer_candidates(cleaned)
        if not extracted:
            extracted = [cleaned]
        for candidate in extracted:
            normalized = clean_benchmark_text(candidate)
            if normalized and normalized not in candidates:
                candidates.append(normalized)
    return candidates


def _strip_answer_prefixes(text):
    cleaned = clean_benchmark_text(text)
    if not cleaned:
        return None
    cleaned = cleaned.replace("**", "").replace("__", "").replace("`", "")
    cleaned = re.sub(r"^\s*(?:final answer|the correct answer is|the answer is|answer)\s*:\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^\s*option\s+[A-E][\)\.\:\s-]*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^\s*[A-E][\)\.\:\s-]+", "", cleaned)
    return cleaned.strip()


def _display_numeric_info(text):
    for candidate in _candidate_texts(text):
        stripped = _strip_answer_prefixes(candidate)
        if not stripped:
            continue
        stripped = stripped.replace(",", "").replace("$", "").strip().strip(".")
        stripped = re.sub(r"^\s*(?:about|approximately|approx\.?|roughly|nearly)\s+", "", stripped, flags=re.IGNORECASE)
        match = re.fullmatch(
            r"(?:[A-Za-z_ ()\-]+:\s*)?([+-]?\d+(?:\.\d+)?)(?:\s*[A-Za-z%°²³]+(?:/[A-Za-z%°²³]+)?)?",
            stripped,
        )
        if not match:
            continue
        number_text = match.group(1)
        decimal_places = len(number_text.split(".", 1)[1]) if "." in number_text else 0
        return {
            "candidate": candidate,
            "number_text": number_text,
            "decimal_places": decimal_places,
            "decimal": Decimal(number_text),
        }
    return None


def _quantize_numeric_value(value, decimal_places):
    try:
        decimal_value = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        return None

    quantum = Decimal("1") if decimal_places <= 0 else Decimal("1").scaleb(-decimal_places)
    try:
        return decimal_value.quantize(quantum, rounding=ROUND_HALF_UP)
    except InvalidOperation:
        return None


def _numeric_values_equivalent(left, right):
    left_numeric = _numeric_value(left)
    right_numeric = _numeric_value(right)
    if left_numeric is None or right_numeric is None:
        return False

    if math.isclose(left_numeric, right_numeric, rel_tol=1e-9, abs_tol=1e-9):
        return True

    left_display = _display_numeric_info(left)
    if left_display and left_display["decimal_places"] > 0:
        rounded_right = _quantize_numeric_value(right_numeric, left_display["decimal_places"])
        if rounded_right is not None and rounded_right == left_display["decimal"]:
            return True

    right_display = _display_numeric_info(right)
    if right_display and right_display["decimal_places"] > 0:
        rounded_left = _quantize_numeric_value(left_numeric, right_display["decimal_places"])
        if rounded_left is not None and rounded_left == right_display["decimal"]:
            return True

    return False


def _normalize_numeric_candidate(text):
    for candidate in _candidate_texts(text):
        cleaned = clean_benchmark_text(candidate)
        if not cleaned:
            continue
        word_number = number_words_to_numeric_string(cleaned)
        if word_number is not None:
            cleaned = word_number
        cleaned = cleaned.replace(",", "")
        cleaned = cleaned.replace("$", "")
        cleaned = cleaned.strip().strip(".")
        if not cleaned:
            continue
        try:
            number = Decimal(cleaned)
        except InvalidOperation:
            continue
        return number.normalize()
    return None


def _gsm_numeric_candidates(*values):
    candidates = []
    seen = set()
    for candidate in _candidate_texts(*values):
        normalized = _normalize_numeric_candidate(candidate)
        if normalized is None:
            extracted_numeric = _extract_gsm_number(candidate)
            if extracted_numeric is not None:
                normalized = _normalize_numeric_candidate(extracted_numeric)
        if normalized is None:
            continue
        key = (candidate, str(normalized))
        if key in seen:
            continue
        seen.add(key)
        candidates.append((candidate, normalized))
    return candidates


def _read_latex_argument(text, start_idx):
    idx = start_idx
    while idx < len(text) and text[idx].isspace():
        idx += 1
    if idx >= len(text):
        return None, idx
    if text[idx] != "{":
        return text[idx], idx + 1

    depth = 1
    idx += 1
    collected = []
    while idx < len(text):
        char = text[idx]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return "".join(collected), idx + 1
        collected.append(char)
        idx += 1
    return None, start_idx


def _latex_to_sympy_expression(text):
    cleaned = clean_benchmark_text(text)
    if not cleaned:
        return None

    cleaned = cleaned.replace("\\left", "")
    cleaned = cleaned.replace("\\right", "")
    cleaned = cleaned.replace("\\cdot", "*")
    cleaned = cleaned.replace("\\times", "*")
    cleaned = cleaned.replace("\\div", "/")
    cleaned = cleaned.replace("\\pi", "pi")
    cleaned = cleaned.replace("×", "*")
    cleaned = cleaned.replace("÷", "/")
    cleaned = cleaned.replace("^", "**")

    converted = []
    idx = 0
    while idx < len(cleaned):
        if cleaned.startswith("\\frac", idx):
            idx += len("\\frac")
            numerator, idx = _read_latex_argument(cleaned, idx)
            denominator, idx = _read_latex_argument(cleaned, idx)
            if numerator is None or denominator is None:
                return None
            converted.append(f"(({_latex_to_sympy_expression(numerator)})/({_latex_to_sympy_expression(denominator)}))")
            continue
        if cleaned.startswith("\\sqrt", idx):
            idx += len("\\sqrt")
            radicand, idx = _read_latex_argument(cleaned, idx)
            if radicand is None:
                return None
            converted.append(f"sqrt(({_latex_to_sympy_expression(radicand)}))")
            continue
        if cleaned[idx] == "{":
            converted.append("(")
            idx += 1
            continue
        if cleaned[idx] == "}":
            converted.append(")")
            idx += 1
            continue
        if cleaned[idx] == "\\":
            command = re.match(r"\\([A-Za-z]+)", cleaned[idx:])
            if not command:
                idx += 1
                continue
            token = command.group(1)
            converted.append({"pm": "+-", "infty": "oo"}.get(token, token))
            idx += len(token) + 1
            continue
        converted.append(cleaned[idx])
        idx += 1

    return "".join(converted)


def _normalize_math_text(text):
    if text is None:
        return None

    cleaned = clean_benchmark_text(text)
    replacements = {
        "\n": "",
        " ": "",
        "\\left": "",
        "\\right": "",
        "\\!": "",
        "\\$": "",
        "$": "",
        "tfrac": "frac",
        "dfrac": "frac",
        "^\\circ": "",
        "^{\\circ}": "",
    }
    for old, new in replacements.items():
        cleaned = cleaned.replace(old, new)
    if cleaned.startswith("\\(") and cleaned.endswith("\\)"):
        cleaned = cleaned[2:-2]
    if cleaned.startswith("\\[") and cleaned.endswith("\\]"):
        cleaned = cleaned[2:-2]
    previous = None
    while cleaned != previous:
        previous = cleaned
        cleaned = (
            cleaned
            .replace("++", "+")
            .replace("+-", "-")
            .replace("-+", "-")
            .replace("--", "+")
        )
    return cleaned.strip(".")


def _normalize_option_text(text):
    if text is None:
        return None

    cleaned = clean_benchmark_text(text)
    replacements = {
        "\n": " ",
        "**": "",
        "__": "",
        "`": "",
        "\\left": "",
        "\\right": "",
        "\\!": "",
        "$": "",
        "Ã‚Â²": "²",
        "Ã‚Â³": "³",
        "Ã¢â‚¬â„¢": "'",
        "Ã¢â‚¬Å“": '"',
        "Ã¢â‚¬Â": '"',
        "Ã¢â‚¬â€œ": "-",
        "Ã¢â‚¬â€": "-",
        "Ã¢ÂˆÂ’": "-",
        "Ã¢Â‰Â¤": "≤",
        "Ã¢Â‰Â¥": "≥",
        "Ã¢ÂˆÂš": "√",
        "−": "-",
        "–": "-",
        "—": "-",
    }
    for old, new in replacements.items():
        cleaned = cleaned.replace(old, new)

    cleaned = re.sub(r"^\s*(?:final answer|the correct answer is|the answer is|answer)\s*:\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^\s*option\s+[A-E][\)\.\:\s-]*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^\s*[A-E][\)\.\:\s-]+", "", cleaned)
    cleaned = re.sub(r"\s+", "", cleaned.lower())
    return cleaned.strip(".")


def _option_entries(options):
    entries = []
    for idx, option in enumerate(options or []):
        if isinstance(option, dict):
            key = clean_benchmark_text(option.get("key")) or chr(ord("A") + idx)
            label = clean_benchmark_text(option.get("label"))
        else:
            option_text = clean_benchmark_text(option)
            match = re.match(r"\s*([A-E])[\)\.\:]?\s*(.*)", option_text or "")
            if match:
                key = match.group(1).upper()
                label = clean_benchmark_text(match.group(2))
            else:
                key = chr(ord("A") + idx)
                label = option_text
        if key and label:
            entries.append((str(key).strip().upper(), label))
    return entries


def _option_value_for_key(option_key, options):
    normalized_key = clean_benchmark_text(option_key)
    if not normalized_key:
        return None
    normalized_key = normalized_key.upper()
    for key, label in _option_entries(options):
        if key == normalized_key:
            return label
    return None


def _numeric_expression_value(text):
    cleaned = clean_benchmark_text(text)
    if not cleaned or parse_expr is None or N is None:
        return None

    candidate = cleaned
    candidate = re.sub(r"^\s*(?:final answer|the correct answer is|the answer is|answer)\s*:\s*", "", candidate, flags=re.IGNORECASE)
    candidate = re.sub(r"^\s*option\s+[A-E][\)\.\:\s-]*", "", candidate, flags=re.IGNORECASE)
    candidate = re.sub(r"^\s*[A-E][\)\.\:\s-]+", "", candidate)
    candidate = candidate.strip().strip(".")
    if not candidate:
        return None

    lowered = candidate.lower()
    if lowered in {"none of these", "none"}:
        return None

    candidate = re.sub(r"√\s*\(([^)]+)\)", r"sqrt(\1)", candidate)
    candidate = re.sub(r"√\s*([A-Za-z0-9]+)", r"sqrt(\1)", candidate)

    replacements = {
        "^": "**",
        "−": "-",
        "–": "-",
        "—": "-",
        "×": "*",
        "÷": "/",
        ",": "",
    }
    for old, new in replacements.items():
        candidate = candidate.replace(old, new)

    try:
        transformations = standard_transformations + (implicit_multiplication_application, convert_xor)
        parsed = parse_expr(candidate, transformations=transformations, evaluate=True)
        return float(N(parsed, 15))
    except Exception:
        return None


def _parse_symbolic_math_candidate(text):
    cleaned = clean_benchmark_text(text)
    if not cleaned or parse_expr is None:
        return None

    candidate = re.sub(
        r"^\s*(?:final answer|the correct answer is|the answer is|answer)\s*:\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    candidate = candidate.strip().strip(".")
    if not candidate:
        return None

    candidate = _latex_to_sympy_expression(candidate) or candidate
    replacements = {
        "^": "**",
        "âˆ’": "-",
        "â€“": "-",
        "â€”": "-",
        "Ã—": "*",
        "Ã·": "/",
    }
    for old, new in replacements.items():
        candidate = candidate.replace(old, new)

    try:
        transformations = standard_transformations + tuple(
            transform
            for transform in (implicit_multiplication_application, convert_xor)
            if transform is not None
        )
        return parse_expr(candidate, transformations=transformations, evaluate=True)
    except Exception:
        return None


def _symbolic_math_equivalent(left, right):
    if parse_expr is None:
        return False

    left_expr = _parse_symbolic_math_candidate(left)
    right_expr = _parse_symbolic_math_candidate(right)
    if left_expr is None or right_expr is None:
        return False

    try:
        difference = left_expr - right_expr
    except Exception:
        difference = None

    if difference is not None:
        try:
            if simplify is not None and simplify(difference) == 0:
                return True
        except Exception:
            pass
        try:
            if difference.equals(0):
                return True
        except Exception:
            pass

    try:
        if left_expr.equals(right_expr):
            return True
    except Exception:
        pass

    return False


def _semantic_numeric_expression_value(text):
    if parse_expr is None or N is None:
        return None

    for raw_candidate in _candidate_texts(text):
        candidate = clean_benchmark_text(raw_candidate)
        candidate = re.sub(r"^\s*(?:final answer|the correct answer is|the answer is|answer)\s*:\s*", "", candidate, flags=re.IGNORECASE)
        candidate = re.sub(r"^\s*option\s+[A-E][\)\.\:\s-]*", "", candidate, flags=re.IGNORECASE)
        candidate = re.sub(r"^\s*[A-E][\)\.\:\s-]+", "", candidate)
        candidate = candidate.strip().strip(".")
        if not candidate:
            continue

        lowered = candidate.lower()
        if lowered in {"none of these", "none"}:
            continue

        candidate = _latex_to_sympy_expression(candidate) or candidate
        candidate = re.sub(r"âˆš\s*\(([^)]+)\)", r"sqrt(\1)", candidate)
        candidate = re.sub(r"âˆš\s*([A-Za-z0-9]+)", r"sqrt(\1)", candidate)

        replacements = {
            "âˆ’": "-",
            "â€“": "-",
            "â€”": "-",
            "Ã—": "*",
            "Ã·": "/",
            ",": "",
        }
        for old, new in replacements.items():
            candidate = candidate.replace(old, new)

        try:
            transformations = standard_transformations + (implicit_multiplication_application, convert_xor)
            parsed = parse_expr(candidate, transformations=transformations, evaluate=True)
            return float(N(parsed, 15))
        except Exception:
            continue
    return None


def _numeric_value(text):
    expression_value = _semantic_numeric_expression_value(text)
    if expression_value is not None:
        return expression_value

    numeric_text = _extract_gsm_number(text)
    normalized_numeric = _normalize_numeric_candidate(numeric_text)
    if normalized_numeric is None:
        return None
    try:
        return float(normalized_numeric)
    except (TypeError, ValueError):
        return None


def _resolve_option_value(text, options):
    cleaned = clean_benchmark_text(text)
    if not cleaned:
        return None

    stripped = cleaned.replace("**", "").replace("__", "").replace("`", "")
    option_letter = _extract_option_letter(stripped)
    if option_letter and options:
        option_label = _option_value_for_key(option_letter, options)
        if option_label:
            return option_label

    if options:
        normalized_candidate = _normalize_option_text(stripped)
        numeric_candidate = _numeric_value(stripped)
        for key, label in _option_entries(options):
            if normalized_candidate == _normalize_option_text(label):
                return label
            prefixed = f"{key}. {label}"
            if normalized_candidate == _normalize_option_text(prefixed):
                return label
            numeric_label = _numeric_value(label)
            if numeric_candidate is not None and numeric_label is not None and _numeric_values_equivalent(stripped, label):
                return label

    return stripped


def _resolved_option_candidates(values, options):
    resolved = []
    for candidate in _candidate_texts(*values):
        if options:
            value = _resolve_option_value(candidate, options)
        else:
            value = _extract_option_letter(candidate) or _normalize_option_text(candidate)
        normalized = clean_benchmark_text(value)
        if normalized and normalized not in {item[1] for item in resolved}:
            resolved.append((candidate, normalized))
    return resolved


def _numeric_candidates(*values):
    candidates = []
    for candidate in _candidate_texts(*values):
        numeric = _numeric_value(candidate)
        if numeric is None:
            normalized = _normalize_numeric_candidate(candidate)
            if normalized is not None:
                try:
                    numeric = float(normalized)
                except (TypeError, ValueError):
                    numeric = None
        if numeric is None:
            continue
        candidates.append((candidate, numeric))
    return candidates


def _first_candidate(*values):
    candidates = _candidate_texts(*values)
    return candidates[0] if candidates else None


def _math_match_pair(left_values, right_values):
    left_candidates = _candidate_texts(*left_values)
    right_candidates = _candidate_texts(*right_values)

    for left_candidate in left_candidates:
        left_math = _normalize_math_text(left_candidate)
        if not left_math:
            continue
        for right_candidate in right_candidates:
            right_math = _normalize_math_text(right_candidate)
            if right_math and left_math == right_math:
                return left_candidate, right_candidate, "math-normalized"

    for left_candidate in left_candidates:
        for right_candidate in right_candidates:
            if _symbolic_math_equivalent(left_candidate, right_candidate):
                return left_candidate, right_candidate, "math-symbolic"

    right_numeric_candidates = _numeric_candidates(*right_values)
    for left_candidate, left_numeric in _numeric_candidates(*left_values):
        for right_candidate, right_numeric in right_numeric_candidates:
            if _numeric_values_equivalent(left_candidate, right_candidate):
                return left_candidate, right_candidate, "math-numeric"

    return None


def _generic_match_pair(left_values, right_values):
    match = _math_match_pair(left_values, right_values)
    if match:
        return match

    left_candidates = _candidate_texts(*left_values)
    right_candidates = _candidate_texts(*right_values)
    normalized_right = {(_normalize_option_text(candidate), candidate) for candidate in right_candidates}
    for left_candidate in left_candidates:
        left_normalized = _normalize_option_text(left_candidate)
        if not left_normalized:
            continue
        for right_normalized, right_candidate in normalized_right:
            if left_normalized == right_normalized:
                return left_candidate, right_candidate, "generic-normalized"

    return None


def answers_match(dataset, left, right, *, options=None):
    if left in (None, "") or right in (None, ""):
        return False

    dataset = dataset_label(dataset)

    if dataset in {"AQUA", "MMLU"}:
        left_resolved = _resolved_option_candidates([left], options)
        right_resolved = _resolved_option_candidates([right], options)
        for _, left_value in left_resolved:
            for _, right_value in right_resolved:
                left_numeric = _numeric_value(left_value)
                right_numeric = _numeric_value(right_value)
                if left_numeric is not None and right_numeric is not None:
                    if _numeric_values_equivalent(left_value, right_value):
                        return True
                elif _normalize_option_text(left_value) == _normalize_option_text(right_value):
                    return True
        return False

    if dataset == "GSM":
        left_number = _normalize_numeric_candidate(left)
        right_number = _normalize_numeric_candidate(right)
        return left_number is not None and right_number is not None and left_number == right_number

    if dataset == "MATH":
        return _math_match_pair([left], [right]) is not None

    return _generic_match_pair([left], [right]) is not None


def _math_candidates(record):
    gold = resolve_gold_answer_bundle(record)
    return _candidate_texts(gold.get("value") or gold.get("display"))


def _raw_answer_hint(value):
    cleaned = clean_benchmark_text(value)
    if not cleaned:
        return None
    lines = [line.strip() for line in str(cleaned).splitlines() if line.strip()]
    return lines[-1] if lines else cleaned


def _raw_predicted_answer(record):
    return _raw_answer_hint(
        record.get("final_answer")
        or record.get("answer")
        or record.get("final_generated_solution")
        or record.get("solution")
        or record.get("solution_generator_output")
        or record.get("program_output")
        or record.get("program_executor:output")
        or record.get("wolfram_output")
        or record.get("wolfram_alpha_search:output")
    )


def _raw_gold_answer(record):
    return _raw_answer_hint(
        record.get("gold_answer")
        or record.get("ground_truth_solution")
        or record.get("correct_option")
    )


def evaluate_record(record):
    dataset = dataset_label(record.get("dataset"))
    predicted = resolve_final_answer_bundle(record)
    gold = resolve_gold_answer_bundle(record)
    predicted_answer = predicted.get("value") or predicted.get("display")
    gold_answer = gold.get("value") or gold.get("display")
    options = record.get("options")

    evaluation = {
        "predicted_answer": predicted_answer,
        "gold_answer": gold_answer,
        "predicted_answer_display": predicted.get("display") or predicted_answer,
        "gold_answer_display": gold.get("display") or gold_answer,
        "predicted_answer_raw": _raw_predicted_answer(record),
        "gold_answer_raw": _raw_gold_answer(record),
        "predicted_answer_option": predicted.get("option_key"),
        "gold_answer_option": gold.get("option_key"),
        "evaluation_status": "not-evaluated",
        "is_correct": None,
        "evaluation_method": None,
        "evaluation_error": None,
    }

    def set_mismatch_error():
        raw_pred = evaluation.get("predicted_answer_raw") or evaluation.get("predicted_answer_display") or evaluation.get("predicted_answer")
        raw_gold = evaluation.get("gold_answer_raw") or evaluation.get("gold_answer_display") or evaluation.get("gold_answer")
        evaluation["evaluation_error"] = f"Raw answer mismatch: predicted '{raw_pred}' vs gold '{raw_gold}'."

    if dataset in {"AQUA", "MMLU"}:
        predicted_values = _resolved_option_candidates([evaluation["predicted_answer_display"] or predicted_answer], options)
        gold_values = _resolved_option_candidates([evaluation["gold_answer_display"] or gold_answer], options)
        evaluation["predicted_answer"] = predicted_values[0][1] if predicted_values else (_first_candidate(predicted_answer) or predicted_answer)
        evaluation["gold_answer"] = gold_values[0][1] if gold_values else (_first_candidate(gold_answer) or gold_answer)
        evaluation["evaluation_method"] = "option-value" if options else "option-letter"
        if not gold_values:
            return evaluation
        if not predicted_values:
            return evaluation
        for _, predicted_value in predicted_values:
            for _, gold_value in gold_values:
                predicted_numeric = _numeric_value(predicted_value)
                gold_numeric = _numeric_value(gold_value)
                if predicted_numeric is not None and gold_numeric is not None:
                    if _numeric_values_equivalent(predicted_value, gold_value):
                        evaluation["predicted_answer"] = predicted_value
                        evaluation["gold_answer"] = gold_value
                        evaluation["evaluation_status"] = "evaluated"
                        evaluation["is_correct"] = True
                        return evaluation
                elif _normalize_option_text(predicted_value) == _normalize_option_text(gold_value):
                    evaluation["predicted_answer"] = predicted_value
                    evaluation["gold_answer"] = gold_value
                    evaluation["evaluation_status"] = "evaluated"
                    evaluation["is_correct"] = True
                    return evaluation
        evaluation["evaluation_status"] = "evaluated"
        evaluation["is_correct"] = False
        set_mismatch_error()
        return evaluation

    if dataset == "GSM":
        predicted_numeric_candidates = _gsm_numeric_candidates(predicted_answer)
        gold_numeric_candidates = _gsm_numeric_candidates(gold_answer)

        evaluation["predicted_answer"] = predicted_numeric_candidates[0][0] if predicted_numeric_candidates else (_first_candidate(predicted_answer) or predicted_answer)
        evaluation["gold_answer"] = gold_numeric_candidates[0][0] if gold_numeric_candidates else (_first_candidate(gold_answer) or gold_answer)
        evaluation["evaluation_method"] = "gsm-numeric"
        if not gold_numeric_candidates:
            return evaluation
        if not predicted_numeric_candidates:
            return evaluation
        for predicted_candidate, normalized_pred in predicted_numeric_candidates:
            for gold_candidate, normalized_gold in gold_numeric_candidates:
                if normalized_pred == normalized_gold:
                    evaluation["predicted_answer"] = predicted_candidate
                    evaluation["gold_answer"] = gold_candidate
                    evaluation["evaluation_status"] = "evaluated"
                    evaluation["is_correct"] = True
                    return evaluation
        evaluation["evaluation_status"] = "evaluated"
        evaluation["is_correct"] = False
        set_mismatch_error()
        return evaluation

    if dataset == "MATH":
        match = _math_match_pair([predicted_answer], _math_candidates(record))
        evaluation["predicted_answer"] = _first_candidate(predicted_answer) or predicted_answer
        evaluation["gold_answer"] = _first_candidate(*_math_candidates(record)) or gold_answer
        evaluation["evaluation_method"] = "math-normalized"
        if match:
            evaluation["predicted_answer"] = match[0]
            evaluation["gold_answer"] = match[1]
            evaluation["evaluation_method"] = match[2]
            evaluation["evaluation_status"] = "evaluated"
            evaluation["is_correct"] = True
            return evaluation

        if evaluation["gold_answer"] and evaluation["predicted_answer"]:
            evaluation["evaluation_status"] = "evaluated"
            evaluation["is_correct"] = False
            set_mismatch_error()
        return evaluation

    match = _generic_match_pair([predicted_answer], [gold_answer])
    evaluation["predicted_answer"] = _first_candidate(predicted_answer) or predicted_answer
    evaluation["gold_answer"] = _first_candidate(gold_answer) or gold_answer
    evaluation["evaluation_method"] = "generic-normalized"
    if match:
        evaluation["predicted_answer"] = match[0]
        evaluation["gold_answer"] = match[1]
        evaluation["evaluation_method"] = match[2]
        evaluation["evaluation_status"] = "evaluated"
        evaluation["is_correct"] = True
        return evaluation

    if evaluation["gold_answer"] and evaluation["predicted_answer"]:
        evaluation["evaluation_status"] = "evaluated"
        evaluation["is_correct"] = False
        set_mismatch_error()
        return evaluation
    return evaluation


def summarize_accuracy(records):
    evaluated = [record for record in records if record.get("evaluation_status") == "evaluated" and record.get("is_correct") is not None]
    correct = sum(1 for record in evaluated if record.get("is_correct"))
    total = len(evaluated)
    accuracy = (correct / total) if total else None
    return {
        "evaluated": total,
        "correct": correct,
        "accuracy": accuracy,
    }


def _timestamp():
    return datetime.now().isoformat(timespec="seconds")


def build_run_benchmark_row(args, records):
    summary = summarize_accuracy(records)
    dataset = dataset_label(args.dataset if args.dataset != "ALL" else "MIXED")
    return {
        "timestamp": _timestamp(),
        "label": args.label,
        "model": args.model,
        "dataset": dataset,
        "split": args.test_split,
        "evaluated": summary["evaluated"],
        "correct": summary["correct"],
        "accuracy": summary["accuracy"],
        "total_records": len(records),
    }


def append_run_benchmark(output_root, row):
    os.makedirs(output_root, exist_ok=True)
    path = os.path.join(output_root, "benchmark_runs.jsonl")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(row, handle, ensure_ascii=False)
        handle.write("\n")
    return path


def update_aggregate_benchmark(output_root, row):
    os.makedirs(output_root, exist_ok=True)
    path = os.path.join(output_root, "benchmark_summary.csv")
    latest = {
        "model": row["model"],
        "dataset": row["dataset"],
        "run_count": 1,
        "total_evaluated": int(row["evaluated"]),
        "total_correct": int(row["correct"]),
        "accuracy": row["accuracy"],
        "last_label": row["label"],
        "last_updated": row["timestamp"],
    }

    fieldnames = [
        "model",
        "dataset",
        "run_count",
        "total_evaluated",
        "total_correct",
        "accuracy",
        "last_label",
        "last_updated",
    ]
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(latest)

    return path, latest

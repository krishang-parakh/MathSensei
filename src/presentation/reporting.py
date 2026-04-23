import argparse
import io
import json
import os
import re
import ast
import tokenize
from collections import Counter
from html import escape

from core.answer_parsing import (
    extract_tagged_answer,
)
from core.answer_resolution import (
    format_option_answer,
    resolve_final_answer_bundle,
    resolve_gold_answer_bundle,
    uses_option_answers,
)
from presentation.asy_rendering import render_asy_block_html, split_asy_blocks
from presentation.benchmarking import (
    _extract_option_letter,
    _normalize_math_text,
    _latex_to_sympy_expression,
    evaluate_record,
    answers_match,
    summarize_accuracy,
)

try:
    from sympy import latex as sympy_latex
    from sympy.parsing.sympy_parser import (
        convert_xor,
        implicit_multiplication_application,
        parse_expr,
        standard_transformations,
    )
except Exception:
    sympy_latex = None
    parse_expr = None
    standard_transformations = ()
    implicit_multiplication_application = None
    convert_xor = None


MODULE_TITLE_MAP = {
    "knowledge_retrieval": "Knowledge Representation",
    "bing_search": "Bing Search",
    "wolfram_alpha_search": "Wolfram Alpha Search",
    "program_generator": "Program Generator",
    "python_generator_refine_executor": "Program Generator + Refine",
    "program_executor": "Program Executor",
    "solution_generator": "Solution Generator",
    "answer_generator": "Answer Generator",
    "query_generator": "Query Generator",
    "wiki_search": "Wikipedia Search",
}

TOP_LEVEL_MODULE_ORDER = [
    "knowledge_retrieval",
    "bing_search",
    "wolfram_alpha_search",
    "program_generator",
    "python_generator_refine_executor",
    "program_executor",
    "solution_generator",
    "answer_generator",
]

DATASET_DASHBOARD_ORDER = ["AQUA", "GSM", "MATH", "MMLU"]


def clean_display_text(text):
    if text is None:
        return None

    cleaned = str(text)
    replacements = {
        "Â²": "²",
        "Â³": "³",
        "â€™": "'",
        "â€œ": '"',
        "â€": '"',
        "â€“": "-",
        "â€”": "-",
        "â": "-",
        "â¤": "≤",
        "â¥": "≥",
        "â": "√",
    }
    for source, target in replacements.items():
        cleaned = cleaned.replace(source, target)
    return cleaned


def _stringify_detail_value(value):
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return clean_display_text(json.dumps(value, indent=2, ensure_ascii=False))
    return clean_display_text(value)


def _coerce_mapping(value):
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        text = clean_display_text(value).strip()
        if not text:
            return {}
        for loader in (json.loads, ast.literal_eval):
            try:
                parsed = loader(text)
            except Exception:
                continue
            if isinstance(parsed, dict):
                return parsed
    return {}


def _normalize_seconds(value):
    if value in (None, ""):
        return None
    if isinstance(value, (int, float)):
        return round(float(value), 4)

    text = clean_display_text(value).strip().lower()
    if not text:
        return None
    if text.endswith("s"):
        text = text[:-1].strip()
    try:
        return round(float(text), 4)
    except ValueError:
        return None


def _normalize_text_mapping(value):
    normalized = {}
    for key, raw_value in _coerce_mapping(value).items():
        normalized_key = _normalize_module_name(key) or clean_display_text(key)
        normalized_value = clean_display_text(raw_value)
        if normalized_key and normalized_value not in (None, ""):
            normalized[normalized_key] = normalized_value
    return normalized


def _normalize_seconds_mapping(value):
    normalized = {}
    for key, raw_value in _coerce_mapping(value).items():
        normalized_key = _normalize_module_name(key) or clean_display_text(key)
        normalized_value = _normalize_seconds(raw_value)
        if normalized_key and normalized_value is not None:
            normalized[normalized_key] = normalized_value
    return normalized


def _normalize_wolfram_trace(trace):
    normalized = []
    for index, step in enumerate(trace or [], start=1):
        if not isinstance(step, dict):
            continue
        normalized_step = {
            "step": step.get("step") or index,
            "thought": clean_display_text(step.get("thought")),
            "query": clean_display_text(step.get("query")),
            "output": clean_display_text(step.get("output")),
            "error": clean_display_text(step.get("error")),
            "source": clean_display_text(step.get("source")),
            "completion_thought": clean_display_text(step.get("completion_thought")),
        }
        if any(value not in (None, "") for key, value in normalized_step.items() if key != "step"):
            normalized.append(normalized_step)
    return normalized


def _normalize_module_name(module_name):
    if module_name is None:
        return None
    cleaned = clean_display_text(module_name).strip()
    if cleaned.startswith("solver."):
        cleaned = cleaned.split(".", 1)[1]
    return cleaned


def _normalize_modules_list(modules):
    if not modules:
        return []
    if isinstance(modules, str):
        text = clean_display_text(modules).strip()
        if not text:
            return []
        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = ast.literal_eval(text)
            except Exception:
                parsed = [segment.strip().strip("'\"") for segment in text[1:-1].split(",") if segment.strip()]
            modules = parsed
        else:
            modules = [segment.strip() for segment in text.split(",") if segment.strip()]

    normalized = []
    for module_name in modules:
        cleaned = _normalize_module_name(module_name)
        if cleaned and cleaned not in normalized:
            normalized.append(cleaned)
    return normalized


def _order_modules(module_names):
    preferred = [name for name in TOP_LEVEL_MODULE_ORDER if name in module_names]
    extras = sorted(name for name in module_names if name not in preferred)
    return preferred + extras


def _infer_modules_from_record(record):
    discovered = []
    for key in record:
        if ":" not in key:
            continue
        prefix, suffix = key.split(":", 1)
        if suffix not in {"input", "output", "error"}:
            continue
        prefix = _normalize_module_name(prefix)
        if prefix == "query_generator":
            continue
        if prefix and prefix not in discovered:
            discovered.append(prefix)

    if record.get("program") and "program_generator" not in discovered and "python_generator_refine_executor" not in discovered:
        discovered.append("program_generator")
    if record.get("program_executor:output") is not None or record.get("program_executor:error"):
        if "program_executor" not in discovered:
            discovered.append("program_executor")
    if record.get("knowledge_retrieval_input") or record.get("knowledge_retrieval_output"):
        if "knowledge_retrieval" not in discovered:
            discovered.append("knowledge_retrieval")
    if record.get("bing_search_input") or record.get("bing_search_output"):
        if "bing_search" not in discovered:
            discovered.append("bing_search")
    if record.get("wolfram_query") or record.get("wolfram_output"):
        if "wolfram_alpha_search" not in discovered:
            discovered.append("wolfram_alpha_search")
    if record.get("program_generator_input") or record.get("program_generator_output"):
        if "program_generator" not in discovered:
            discovered.append("program_generator")
    if record.get("program_output") is not None or record.get("program_error"):
        if "program_executor" not in discovered:
            discovered.append("program_executor")
    if record.get("solution_generator_input") or record.get("solution_generator_output"):
        if "solution_generator" not in discovered:
            discovered.append("solution_generator")
    if record.get("solution") and "solution_generator" not in discovered:
        discovered.append("solution_generator")

    return _order_modules(discovered)


def _module_title(module_name):
    normalized = _normalize_module_name(module_name) or "module"
    return MODULE_TITLE_MAP.get(normalized, normalized.replace("_", " ").title())


def _module_field_value(record, module_name, field_name):
    module_name = _normalize_module_name(module_name)
    candidates = [
        f"{module_name}:{field_name}",
        f"{module_name}_{field_name}",
    ]

    alias_map = {
        ("knowledge_retrieval", "input"): ["knowledge_retrieval_input"],
        ("knowledge_retrieval", "output"): ["knowledge_retrieval_output"],
        ("bing_search", "input"): ["bing_search_input"],
        ("bing_search", "output"): ["bing_search_output"],
        ("wolfram_alpha_search", "input"): ["wolfram_query"],
        ("wolfram_alpha_search", "output"): ["wolfram_output"],
        ("program_generator", "input"): ["program_generator_input"],
        ("program_generator", "output"): ["program_generator_output", "generated_program"],
        ("python_generator_refine_executor", "input"): ["program_generator_input", "python_generator_refine_executor_input"],
        ("python_generator_refine_executor", "output"): ["program_generator_output", "generated_program", "python_generator_refine_executor_output"],
        ("program_executor", "output"): ["program_output"],
        ("program_executor", "error"): ["program_error"],
        ("solution_generator", "input"): ["solution_generator_input"],
        ("solution_generator", "output"): ["solution_generator_output", "final_generated_solution"],
    }
    candidates.extend(alias_map.get((module_name, field_name), []))

    if module_name in {"program_generator", "python_generator_refine_executor"} and field_name == "output":
        candidates.append("program")

    seen = set()
    for key in candidates:
        if key in seen:
            continue
        seen.add(key)
        value = record.get(key)
        rendered = _stringify_detail_value(value)
        if rendered not in (None, ""):
            return rendered
    return None


def _module_backend_value(record, module_name):
    module_name = _normalize_module_name(module_name)
    return _normalize_text_mapping(record.get("module_backends")).get(module_name)


def _module_elapsed_seconds(record, module_name):
    module_name = _normalize_module_name(module_name)
    return _normalize_seconds_mapping(record.get("module_timings_seconds")).get(module_name)


def _format_elapsed_seconds(value):
    elapsed = _normalize_seconds(value)
    if elapsed is None:
        return None
    if elapsed < 1:
        return f"{elapsed * 1000:.0f}ms"
    if elapsed < 60:
        return f"{elapsed:.2f}s"
    if elapsed < 3600:
        minutes = int(elapsed // 60)
        seconds = elapsed - (minutes * 60)
        return f"{minutes}m {seconds:04.1f}s"
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = elapsed % 60
    return f"{hours}h {minutes:02d}m {seconds:04.1f}s"


def _module_meta_items(record, module_name):
    meta_items = []
    backend = _module_backend_value(record, module_name)
    elapsed = _module_elapsed_seconds(record, module_name)
    if backend:
        meta_items.append({"label": "Model", "value": backend})
    if elapsed is not None:
        meta_items.append({"label": "Time", "value": _format_elapsed_seconds(elapsed)})
    return meta_items


def _step_item(label, value, code=False, strip_comments=False):
    rendered_value = _stringify_detail_value(value)
    if rendered_value in (None, ""):
        return None
    return {
        "label": label,
        "value": rendered_value,
        "code": code,
        "strip_comments": strip_comments,
    }


def _refine_trace(record):
    rounds = []
    for key in sorted(record.keys()):
        if key.startswith("refine_round"):
            rounds.append({key: record[key]})
    if not rounds:
        return None
    return rounds


def _module_step(record, module_name):
    module_name = _normalize_module_name(module_name)
    items = []
    meta_items = _module_meta_items(record, module_name)

    if module_name == "knowledge_retrieval":
        items.extend(
            [
                _step_item("Reasoning", _module_field_value(record, module_name, "output")),
            ]
        )
    elif module_name == "bing_search":
        items.extend(
            [
                _step_item("Search Output", _module_field_value(record, module_name, "output")),
            ]
        )
    elif module_name == "wolfram_alpha_search":
        wolfram_trace = record.get("wolfram_trace") or record.get("wolfram_alpha_search:trace") or []
        if wolfram_trace:
            for step in wolfram_trace:
                step_number = step.get("step") or len(items) + 1
                items.extend(
                    [
                        _step_item(f"Wolfram Step {step_number} Thought", step.get("thought")),
                        _step_item(f"Wolfram Step {step_number} Query", step.get("query"), code=True),
                        _step_item(f"Wolfram Step {step_number} Output", step.get("output")),
                        _step_item(f"Wolfram Step {step_number} Error", step.get("error"), code=True),
                        _step_item(f"Wolfram Step {step_number} Completion Check", step.get("completion_thought")),
                    ]
                )
        else:
            items.extend(
                [
                    _step_item("Wolfram Query", _module_field_value(record, module_name, "input"), code=True),
                    _step_item("Wolfram Output", _module_field_value(record, module_name, "output")),
                ]
            )
    elif module_name in {"program_generator", "python_generator_refine_executor"}:
        items.extend(
            [
                _step_item("Generated Program", _module_field_value(record, module_name, "output"), code=True),
            ]
        )
    elif module_name == "program_executor":
        repair = record.get("program_repair") or {}
        items.extend(
            [
                _step_item("Executed Program", record.get("generated_program") or _module_field_value(record, "program_generator", "output"), code=True),
                _step_item("Program Output", _module_field_value(record, module_name, "output"), code=True),
                _step_item("Program Error", _module_field_value(record, module_name, "error"), code=True),
                _step_item("Repaired Program", repair.get("repaired_program"), code=True),
                _step_item("Repaired Output", repair.get("repaired_output"), code=True),
                _step_item("Repair Error", repair.get("repaired_error"), code=True),
            ]
        )
    elif module_name == "solution_generator":
        items.extend(
            [
                _step_item("Reasoning and Solution", _module_field_value(record, module_name, "output")),
            ]
        )
    else:
        items.extend(
            [
                _step_item("Output", _module_field_value(record, module_name, "output")),
                _step_item("Error", _module_field_value(record, module_name, "error"), code=True),
            ]
        )

    items = [item for item in items if item]
    if not items and not meta_items:
        return None

    return {
        "module": module_name,
        "title": _module_title(module_name),
        "meta": meta_items,
        "items": items,
    }


def _collect_module_steps(record, modules_used=None):
    module_names = _normalize_modules_list(modules_used or record.get("modules") or record.get("modules_used"))
    if not module_names:
        module_names = _infer_modules_from_record(record)

    steps = []
    for module_name in module_names:
        step = _module_step(record, module_name)
        if step:
            steps.append(step)
    return steps


def _normalize_module_steps(steps):
    normalized_steps = []
    for step in steps or []:
        items = []
        meta = []
        for item in step.get("items", []):
            normalized_item = _step_item(
                item.get("label", "Detail"),
                item.get("value"),
                code=bool(item.get("code")),
                strip_comments=bool(item.get("strip_comments")),
            )
            if normalized_item:
                items.append(normalized_item)
        for meta_item in step.get("meta", []):
            label = clean_display_text(meta_item.get("label"))
            value = clean_display_text(meta_item.get("value"))
            if label and value:
                meta.append({"label": label, "value": value})
        if items or meta:
            normalized_steps.append(
                {
                    "module": _normalize_module_name(step.get("module")),
                    "title": clean_display_text(step.get("title")) or _module_title(step.get("module")),
                    "meta": meta,
                    "items": items,
                }
            )
    return normalized_steps


def _dataset_hint_from_strings(source_path=None, title=None):
    haystacks = [clean_display_text(source_path or ""), clean_display_text(title or "")]
    for haystack in haystacks:
        lowered = haystack.lower()
        if "aqua" in lowered:
            return "AQUA"
        if "mmlu" in lowered:
            return "MMLU"
        if "gsm" in lowered:
            return "GSM"
        if "math" in lowered:
            return "MATH"
    return "UNKNOWN"


def infer_dataset(records=None, source_path=None, title=None):
    hint = _dataset_hint_from_strings(source_path=source_path, title=title)
    if not records:
        return hint

    explicit_datasets = {
        clean_display_text(record.get("dataset"))
        for record in records
        if clean_display_text(record.get("dataset")) not in (None, "", "Unknown", "UNKNOWN")
    }
    if len(explicit_datasets) > 1:
        return "MIXED"
    if len(explicit_datasets) == 1:
        return next(iter(explicit_datasets))

    total = max(len(records), 1)
    option_counts = [len(record.get("options") or []) for record in records]
    option_records = sum(1 for count in option_counts if count > 0)
    four_option_records = sum(1 for count in option_counts if count == 4)
    five_option_records = sum(1 for count in option_counts if count >= 5)
    correct_option_records = sum(1 for record in records if record.get("correct_option"))
    level_records = sum(1 for record in records if record.get("level") not in (None, "", "Unknown"))
    typed_records = sum(1 for record in records if record.get("problem_type") not in (None, "", "Unknown"))
    none_of_these_records = sum(
        1
        for record in records
        if any("none of these" in (option.get("label") or "").lower() for option in (record.get("options") or []))
    )

    option_ratio = option_records / total
    four_option_ratio = four_option_records / total
    five_option_ratio = five_option_records / total
    level_ratio = level_records / total
    typed_ratio = typed_records / total
    correct_ratio = correct_option_records / total

    if option_ratio >= 0.4:
        if five_option_ratio >= four_option_ratio or none_of_these_records > 0:
            return "AQUA"
        if four_option_ratio > 0:
            return "MMLU"
        if hint in {"AQUA", "MMLU"}:
            return hint
        return "AQUA"

    if level_ratio >= 0.2:
        return "MATH"

    if hint != "UNKNOWN":
        if hint == "MATH" and option_ratio >= 0.2 and correct_ratio >= 0.2:
            return "AQUA"
        return hint

    if typed_ratio >= 0.35:
        return "MATH"

    if option_ratio == 0:
        return "GSM"

    return "UNKNOWN"


def infer_final_answer(record):
    resolved = resolve_final_answer_bundle(record)
    return resolved.get("display")


def infer_correct_option(text):
    return _extract_option_letter(text)


def parse_embedded_options(problem_text):
    if not problem_text:
        return None, None

    text = clean_display_text(problem_text)

    aqua_match = re.search(r"^(.*?)(?:\s+Options:\s*)(\[[\s\S]*\])\s*$", text)
    if aqua_match:
        stem = aqua_match.group(1).strip()
        try:
            raw_options = ast.literal_eval(aqua_match.group(2))
        except Exception:
            raw_options = None

        options = []
        if isinstance(raw_options, list):
            for idx, option in enumerate(raw_options):
                option_text = clean_display_text(option)
                letter_match = re.match(r"\s*([A-E])[\)\.:]?\s*(.*)", option_text)
                if letter_match:
                    letter = letter_match.group(1).upper()
                    label = letter_match.group(2).strip()
                else:
                    letter = chr(ord("A") + idx)
                    label = option_text
                options.append({"key": letter, "label": label})
        return stem, options or None

    if "Option A:" in text:
        parts = text.splitlines()
        stem_lines = []
        options = []
        for line in parts:
            option_match = re.match(r"\s*Option\s+([A-D]):\s*(.*)", line)
            if option_match:
                options.append(
                    {
                        "key": option_match.group(1).upper(),
                        "label": option_match.group(2).strip(),
                    }
                )
            else:
                stem_lines.append(line)
        stem = "\n".join(line for line in stem_lines if line.strip()).strip()
        return stem or text, options or None

    return text, None


def normalize_options(raw_options):
    if not raw_options:
        return None

    normalized_options = []
    for idx, option in enumerate(raw_options):
        if isinstance(option, dict):
            key = clean_display_text(option.get("key") or option.get("option") or option.get("id") or chr(ord("A") + idx)).strip()
            label = clean_display_text(option.get("label") or option.get("text") or option.get("value") or "")
        else:
            option_text = clean_display_text(option)
            match = re.match(r"\s*([A-Z])[\)\.:]?\s*(.*)", option_text)
            if match:
                key = match.group(1).upper()
                label = match.group(2).strip()
            else:
                key = chr(ord("A") + idx)
                label = option_text.strip()

        normalized_options.append({"key": key, "label": label})

    return normalized_options or None


def normalize_record(record):
    if "example" in record:
        example = record.get("example", {})
        options = None
        if example.get("options"):
            options = normalize_options(example.get("options"))
        elif any(example.get(f"Option {letter}") for letter in ["A", "B", "C", "D"]):
            options = normalize_options(
                [
                    {"key": "A", "label": example.get("Option A", "")},
                    {"key": "B", "label": example.get("Option B", "")},
                    {"key": "C", "label": example.get("Option C", "")},
                    {"key": "D", "label": example.get("Option D", "")},
                ]
            )

        modules_used = _normalize_modules_list(record.get("modules"))
        if not modules_used:
            modules_used = _infer_modules_from_record(record)

        normalized = {
            "pid": record.get("pid"),
            "dataset": record.get("dataset") or example.get("dataset"),
            "problem": example.get("problem") or example.get("question") or example.get("Question"),
            "problem_type": example.get("type") or example.get("subject"),
            "level": example.get("level"),
            "ground_truth_solution": example.get("solution") or example.get("answer") or example.get("rationale") or example.get("Answer"),
            "gold_answer": example.get("answer") or example.get("Answer") or example.get("correct"),
            "generated_program": record.get("program"),
            "program_output": record.get("program_executor:output"),
            "program_error": record.get("program_executor:error"),
            "wolfram_query": record.get("wolfram_alpha_search:input"),
            "wolfram_output": record.get("wolfram_alpha_search:output"),
            "wolfram_error": record.get("wolfram_alpha_search:error"),
            "wolfram_trace": record.get("wolfram_alpha_search:trace"),
            "knowledge_retrieval_input": record.get("knowledge_retrieval:input"),
            "knowledge_retrieval_output": record.get("knowledge_retrieval:output"),
            "bing_search_input": record.get("bing_search:input"),
            "bing_search_output": record.get("bing_search:output"),
            "program_generator_input": record.get("program_generator:input"),
            "program_generator_output": record.get("program_generator:output"),
            "solution_generator_input": record.get("solution_generator:input"),
            "solution_generator_output": record.get("solution_generator:output"),
            "final_generated_solution": record.get("solution"),
            "final_answer": record.get("answer"),
            "program_repair": record.get("program_repair"),
            "module_warnings": record.get("module_warnings"),
            "module_errors": record.get("module_errors"),
            "module_timings_seconds": record.get("module_timings_seconds"),
            "module_backends": record.get("module_backends"),
            "question_elapsed_seconds": record.get("question_elapsed_seconds"),
            "run_model": record.get("run_model"),
            "question_signature": record.get("question_signature"),
            "method_evaluations": record.get("method_evaluations"),
            "overall_evaluation_label": record.get("overall_evaluation_label"),
            "overall_evaluation_status": record.get("overall_evaluation_status"),
            "overall_is_correct": record.get("overall_is_correct"),
            "modules_used": modules_used,
            "module_steps": _collect_module_steps(record, modules_used),
            "options": options,
            "correct_option": (
                clean_display_text(example.get("correct"))
                or clean_display_text(example.get("Answer"))
                or infer_correct_option(example.get("answer"))
                or infer_correct_option(example.get("Answer"))
                or infer_correct_option(record.get("solution"))
            ),
        }
    else:
        normalized = {
            "pid": record.get("pid"),
            "dataset": record.get("dataset"),
            "problem": record.get("problem"),
            "problem_type": record.get("problem_type"),
            "level": record.get("level"),
            "ground_truth_solution": record.get("ground_truth_solution"),
            "gold_answer": record.get("gold_answer"),
            "generated_program": record.get("generated_program"),
            "program_output": record.get("program_output"),
            "program_error": record.get("program_error"),
            "wolfram_query": record.get("wolfram_query"),
            "wolfram_output": record.get("wolfram_output"),
            "wolfram_error": record.get("wolfram_error"),
            "wolfram_trace": record.get("wolfram_trace"),
            "knowledge_retrieval_input": record.get("knowledge_retrieval_input"),
            "knowledge_retrieval_output": record.get("knowledge_retrieval_output"),
            "bing_search_input": record.get("bing_search_input"),
            "bing_search_output": record.get("bing_search_output"),
            "program_generator_input": record.get("program_generator_input"),
            "program_generator_output": record.get("program_generator_output"),
            "solution_generator_input": record.get("solution_generator_input"),
            "solution_generator_output": record.get("solution_generator_output"),
            "final_generated_solution": record.get("final_generated_solution"),
            "final_answer": record.get("final_answer"),
            "program_repair": record.get("program_repair"),
            "module_warnings": record.get("module_warnings"),
            "module_errors": record.get("module_errors"),
            "module_timings_seconds": record.get("module_timings_seconds"),
            "module_backends": record.get("module_backends"),
            "question_elapsed_seconds": record.get("question_elapsed_seconds"),
            "run_model": record.get("run_model"),
            "question_signature": record.get("question_signature"),
            "method_evaluations": record.get("method_evaluations"),
            "overall_evaluation_label": record.get("overall_evaluation_label"),
            "overall_evaluation_status": record.get("overall_evaluation_status"),
            "overall_is_correct": record.get("overall_is_correct"),
            "modules_used": _normalize_modules_list(record.get("modules_used")),
            "module_steps": _normalize_module_steps(record.get("module_steps")),
            "options": normalize_options(record.get("options")),
            "correct_option": clean_display_text(record.get("correct_option")) or infer_correct_option(record.get("gold_answer")) or infer_correct_option(record.get("ground_truth_solution")),
        }

    modules = normalized.get("modules_used")
    if modules is None:
        normalized["modules_used"] = []
    else:
        normalized["modules_used"] = _normalize_modules_list(modules)
    if not normalized["modules_used"]:
        normalized["modules_used"] = _infer_modules_from_record(normalized)

    normalized["dataset"] = clean_display_text(normalized.get("dataset"))
    normalized["problem_type"] = normalized.get("problem_type") or "Unknown"
    normalized["level"] = normalized.get("level") or "Unknown"
    stem, embedded_options = parse_embedded_options(normalized.get("problem") or "(Problem text unavailable)")
    normalized["problem"] = clean_display_text(stem or "(Problem text unavailable)")
    if not normalized.get("options") and embedded_options:
        normalized["options"] = normalize_options(embedded_options)
    normalized["ground_truth_solution"] = clean_display_text(normalized.get("ground_truth_solution"))
    normalized["gold_answer"] = clean_display_text(normalized.get("gold_answer"))
    normalized["generated_program"] = clean_display_text(normalized.get("generated_program"))
    normalized["program_output"] = clean_display_text(normalized.get("program_output"))
    normalized["program_error"] = clean_display_text(normalized.get("program_error"))
    normalized["wolfram_query"] = clean_display_text(normalized.get("wolfram_query"))
    normalized["wolfram_output"] = clean_display_text(normalized.get("wolfram_output"))
    normalized["wolfram_error"] = clean_display_text(normalized.get("wolfram_error"))
    normalized["wolfram_trace"] = _normalize_wolfram_trace(normalized.get("wolfram_trace"))
    normalized["knowledge_retrieval_input"] = clean_display_text(normalized.get("knowledge_retrieval_input"))
    normalized["knowledge_retrieval_output"] = clean_display_text(normalized.get("knowledge_retrieval_output"))
    normalized["bing_search_input"] = clean_display_text(normalized.get("bing_search_input"))
    normalized["bing_search_output"] = clean_display_text(normalized.get("bing_search_output"))
    normalized["program_generator_input"] = clean_display_text(normalized.get("program_generator_input"))
    normalized["program_generator_output"] = clean_display_text(normalized.get("program_generator_output"))
    normalized["solution_generator_input"] = clean_display_text(normalized.get("solution_generator_input"))
    normalized["solution_generator_output"] = clean_display_text(normalized.get("solution_generator_output"))
    normalized["final_generated_solution"] = clean_display_text(normalized.get("final_generated_solution"))
    normalized["correct_option"] = clean_display_text(normalized.get("correct_option"))
    normalized["module_timings_seconds"] = _normalize_seconds_mapping(normalized.get("module_timings_seconds"))
    normalized["module_backends"] = _normalize_text_mapping(normalized.get("module_backends"))
    normalized["question_elapsed_seconds"] = _normalize_seconds(normalized.get("question_elapsed_seconds"))
    normalized["run_model"] = clean_display_text(normalized.get("run_model"))
    normalized["question_signature"] = clean_display_text(normalized.get("question_signature"))
    normalized["module_warnings"] = [clean_display_text(item) for item in (normalized.get("module_warnings") or []) if clean_display_text(item)]
    normalized["module_errors"] = [clean_display_text(item) for item in (normalized.get("module_errors") or []) if clean_display_text(item)]
    normalized["module_steps"] = normalized.get("module_steps") or _collect_module_steps(normalized, normalized["modules_used"])
    normalized["dataset"] = normalized.get("dataset") or infer_dataset(records=[normalized])
    if normalized["dataset"] in (None, "", "UNKNOWN"):
        normalized["dataset"] = "Unknown"
    evaluation = evaluate_record(normalized)
    normalized.update(evaluation)
    normalized["final_answer"] = evaluation.get("predicted_answer_display")
    normalized["final_answer_value"] = evaluation.get("predicted_answer")
    normalized["final_answer_option"] = evaluation.get("predicted_answer_option")
    normalized["gold_answer"] = evaluation.get("gold_answer")
    normalized["gold_answer_display"] = evaluation.get("gold_answer_display") or evaluation.get("gold_answer")
    if uses_option_answers(normalized):
        normalized["correct_option"] = (
            evaluation.get("gold_answer_option")
            or clean_display_text(normalized.get("correct_option"))
            or infer_correct_option(normalized.get("gold_answer"))
            or infer_correct_option(normalized.get("ground_truth_solution"))
        )
    else:
        normalized["correct_option"] = None
    normalized["method_evaluations"] = _build_method_evaluations(normalized)
    normalized.update(_overall_method_evaluation(normalized))
    normalized["status"] = classify_record(normalized)

    return normalized


def classify_record(record):
    overall_label = record.get("overall_evaluation_label")
    if overall_label == "Correct":
        return "complete"
    if overall_label == "Incorrect":
        return "incorrect-evaluation"
    return "needs-review"


def _record_identity(record):
    signature = clean_display_text(record.get("question_signature"))
    if signature:
        return ("signature", signature)

    pid = record.get("pid")
    dataset = clean_display_text(record.get("dataset"))
    problem = clean_display_text(record.get("problem"))
    if pid is None or problem in (None, ""):
        return None

    return ("pid-problem", dataset or "", str(pid), problem)


def load_records(input_path):
    records = []
    skipped_rows = 0
    seen_identities = {}
    with open(input_path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                normalized = normalize_record(json.loads(line))
            except json.JSONDecodeError:
                skipped_rows += 1
                continue

            identity = _record_identity(normalized)
            if identity is not None and identity in seen_identities:
                records[seen_identities[identity]] = normalized
                continue

            if identity is not None:
                seen_identities[identity] = len(records)
            records.append(normalized)
    return records, skipped_rows


METHOD_SCOREBOARD_ORDER = ["Solution Generator", "Python", "Wolfram"]

METHOD_SOURCE_MAP = {
    "Solution Generator": {
        "modules": {"solution_generator"},
        "fields": {"solution_generator_output", "final_generated_solution", "final_answer"},
    },
    "Python": {
        "modules": {"program_generator", "python_generator_refine_executor", "program_executor"},
        "fields": {"generated_program", "program_output", "program_error"},
    },
    "Wolfram": {
        "modules": {"wolfram_alpha_search"},
        "fields": {"wolfram_query", "wolfram_output", "wolfram_error"},
    },
    "Knowledge": {
        "modules": {"knowledge_retrieval"},
        "fields": {"knowledge_retrieval_input", "knowledge_retrieval_output"},
    },
}

METHOD_STATUS_META = {
    "correct": {"label": "Correct", "symbol": "&#10003;"},
    "incorrect": {"label": "Incorrect", "symbol": "&#10007;"},
    "error": {"label": "Error", "symbol": "!"},
    "incomplete": {"label": "Incomplete", "symbol": "&middot;"},
}

METHOD_FILTER_STATUS_ORDER = ["flagged", "incorrect", "error", "incomplete", "correct"]
METHOD_FILTER_OPTION_LABELS = {
    "flagged": "Flagged",
    "incorrect": "Incorrect",
    "error": "Error",
    "incomplete": "Incomplete",
    "correct": "Correct",
}


def _evaluate_candidate_answer(record, candidate_answer):
    synthetic_record = {
        "dataset": record.get("dataset"),
        "final_answer": candidate_answer,
        "final_generated_solution": candidate_answer,
        "gold_answer": record.get("gold_answer"),
        "ground_truth_solution": record.get("ground_truth_solution"),
        "correct_option": record.get("correct_option"),
        "options": record.get("options"),
    }
    return evaluate_record(synthetic_record)


def _method_filter_slug(label):
    return re.sub(r"[^a-z0-9]+", "-", str(label or "").strip().lower()).strip("-")


def _method_filter_attr_name(label):
    return f"data-{_method_filter_slug(label)}-status"


def _method_error_text(record, label):
    direct_error_map = {
        "Solution Generator": [
            record.get("solution_generator_error"),
        ],
        "Python": [
            record.get("program_error"),
        ],
        "Wolfram": [
            record.get("wolfram_error"),
        ],
        "Knowledge": [
            record.get("knowledge_retrieval_error"),
        ],
    }

    for value in direct_error_map.get(label, []):
        cleaned = clean_display_text(value)
        if cleaned:
            return cleaned

    module_error_prefixes = {
        "Solution Generator": ("solution_generator",),
        "Python": ("program_generator", "python_generator_refine_executor", "program_executor"),
        "Wolfram": ("wolfram_alpha_search",),
        "Knowledge": ("knowledge_retrieval",),
    }
    for error in record.get("module_errors") or []:
        cleaned = clean_display_text(error)
        if not cleaned:
            continue
        lowered = cleaned.lower()
        if any(prefix in lowered for prefix in module_error_prefixes.get(label, ())):
            return cleaned

    return None


def _method_is_relevant(record, label):
    spec = METHOD_SOURCE_MAP.get(label) or {}
    modules_used = set(_normalize_modules_list(record.get("modules_used") or record.get("modules")))
    if modules_used.intersection(spec.get("modules", set())):
        return True
    return any(record.get(field) not in (None, "") for field in spec.get("fields", set()))


def _build_method_evaluations(record):
    answer_candidates = dict(_method_answer_candidates(record))
    evaluations = []

    for label in METHOD_SCOREBOARD_ORDER:
        if not _method_is_relevant(record, label):
            continue

        candidate_answer = answer_candidates.get(label)
        evaluation = _evaluate_candidate_answer(record, candidate_answer) if candidate_answer not in (None, "") else None
        error_text = _method_error_text(record, label)
        if error_text:
            status = "error"
        elif evaluation and evaluation.get("evaluation_status") == "evaluated" and evaluation.get("is_correct") is not None:
            status = "correct" if evaluation.get("is_correct") else "incorrect"
        else:
            status = "incomplete"

        display_answer = None
        if evaluation:
            display_answer = evaluation.get("predicted_answer_display") or evaluation.get("predicted_answer")
        if display_answer in (None, ""):
            display_answer = candidate_answer
        if display_answer in (None, ""):
            display_answer = "No answer extracted"
        if status == "error" and display_answer == "No answer extracted":
            display_answer = error_text

        evaluations.append(
            {
                "label": label,
                "status": status,
                "status_label": METHOD_STATUS_META[status]["label"],
                "symbol": METHOD_STATUS_META[status]["symbol"],
                "answer": clean_display_text(display_answer),
                "is_correct": evaluation.get("is_correct") if evaluation else None,
                "evaluation_status": evaluation.get("evaluation_status") if evaluation else "not-evaluated",
                "has_error": bool(error_text),
            }
        )

    return evaluations


def _overall_method_evaluation(record):
    if not record.get("final_answer"):
        return {
            "overall_evaluation_label": "Needs Review",
            "overall_evaluation_status": "not-evaluated",
            "overall_is_correct": None,
        }

    if record.get("evaluation_status") != "evaluated" or record.get("is_correct") is None:
        return {
            "overall_evaluation_label": "Needs Review",
            "overall_evaluation_status": "not-evaluated",
            "overall_is_correct": None,
        }

    if record.get("is_correct") is False:
        return {
            "overall_evaluation_label": "Incorrect",
            "overall_evaluation_status": "evaluated",
            "overall_is_correct": False,
        }

    return {
        "overall_evaluation_label": "Correct",
        "overall_evaluation_status": "evaluated",
        "overall_is_correct": True,
    }


def summarize_method_accuracy(records):
    scoreboard = []
    for label in METHOD_SCOREBOARD_ORDER:
        available = 0
        evaluated = 0
        correct = 0
        incorrect = 0
        error = 0
        incomplete = 0

        for record in records:
            method_rows = {row["label"]: row for row in (record.get("method_evaluations") or [])}
            row = method_rows.get(label)
            if not row:
                continue
            available += 1
            status = row.get("status")
            if status == "correct":
                correct += 1
            elif status == "incorrect":
                incorrect += 1
            elif status == "error":
                error += 1
            else:
                incomplete += 1
            if row.get("evaluation_status") == "evaluated" and row.get("is_correct") is not None:
                evaluated += 1

        if available == 0:
            continue

        scoreboard.append(
            {
                "label": label,
                "available": available,
                "evaluated": evaluated,
                "correct": correct,
                "incorrect": incorrect,
                "error": error,
                "incomplete": incomplete,
                "flagged": incorrect + error + incomplete,
                "accuracy": (correct / evaluated) if evaluated else None,
            }
        )

    return scoreboard


def build_summary(records):
    total = len(records)
    complete = sum(1 for record in records if record["status"] == "complete")
    needs_review = sum(1 for record in records if record["status"] == "needs-review")
    incorrect_evaluation = sum(1 for record in records if record["status"] == "incorrect-evaluation")
    accuracy = summarize_accuracy(records)

    type_counts = Counter(record["problem_type"] for record in records)
    level_counts = Counter(record["level"] for record in records)
    dataset_counts = Counter(record.get("dataset", "Unknown") for record in records)

    return {
        "total": total,
        "complete": complete,
        "complete_display": f"{complete}/{total}" if total else "0/0",
        "needs_review": needs_review,
        "incorrect_evaluation": incorrect_evaluation,
        "evaluated": accuracy["evaluated"],
        "correct": accuracy["correct"],
        "accuracy": accuracy["accuracy"],
        "method_accuracy": summarize_method_accuracy(records),
        "datasets": dict(sorted(dataset_counts.items(), key=lambda item: (-item[1], item[0]))),
        "types": dict(sorted(type_counts.items(), key=lambda item: (-item[1], item[0]))),
        "levels": dict(sorted(level_counts.items(), key=lambda item: _level_sort_key(item[0]))),
    }


def _distribution_rows(title, values):
    if not values:
        return ""

    rows = []
    max_value = max(values.values())
    for label, count in values.items():
        ratio = 0 if max_value == 0 else int((count / max_value) * 100)
        rows.append(
            f"""
            <div class="distribution-row">
              <div class="distribution-meta">
                <span>{escape(str(label))}</span>
                <strong>{count}</strong>
              </div>
              <div class="distribution-bar">
                <span style="width: {ratio}%"></span>
              </div>
            </div>
            """
        )

    return f"""
    <section class="panel">
      <h3>{escape(title)}</h3>
      <div class="distribution-list">
        {''.join(rows)}
      </div>
    </section>
    """


def _maybe_distribution(title, values):
    if len(values) <= 1:
        return ""
    return _distribution_rows(title, values)


def _extract_level_number(level):
    if level in (None, "", "Unknown"):
        return None
    match = re.search(r"(\d+)", str(level))
    if not match:
        return None
    return int(match.group(1))


def _level_sort_key(level):
    number = _extract_level_number(level)
    if number is None:
        return (1, float("inf"), str(level or "").lower())
    return (0, number, str(level or "").lower())


def _record_sort_key(record):
    return (
        str(record.get("dataset") or "zzzz").lower(),
        str(record.get("problem_type") or "zzzz").lower(),
        _level_sort_key(record.get("level")),
        _display_problem_number(record.get("pid")),
    )


def _sorted_records(records):
    return sorted(records, key=_record_sort_key)


def _sort_filter_values(attr, values):
    values = [value for value in values if value not in (None, "", "Unknown")]
    if attr == "level":
        return sorted(values, key=_level_sort_key)
    return sorted(values, key=lambda value: str(value).lower())


def _filter_options(values):
    options = []
    for value in values:
        if isinstance(value, dict):
            option_value = clean_display_text(value.get("value"))
            option_label = clean_display_text(value.get("label")) or option_value
        else:
            option_value = clean_display_text(value)
            option_label = option_value
        if option_value in (None, "") or option_label in (None, ""):
            continue
        options.append({"value": option_value, "label": option_label})
    return options


def _method_filter_options(records, label):
    statuses = {
        row.get("status")
        for record in records
        for row in (record.get("method_evaluations") or [])
        if row.get("label") == label and row.get("status")
    }
    if not statuses:
        return []

    options = []
    if any(status != "correct" for status in statuses):
        options.append({"value": "flagged", "label": METHOD_FILTER_OPTION_LABELS["flagged"]})
    for status in METHOD_FILTER_STATUS_ORDER:
        if status == "flagged" or status not in statuses:
            continue
        options.append({"value": status, "label": METHOD_FILTER_OPTION_LABELS[status]})
    return options


def _compact_review_text(text, prefer_last_line=False, limit=140):
    if text in (None, ""):
        return None
    cleaned = clean_display_text(text)
    lines = [line.strip() for line in str(cleaned).splitlines() if line.strip()]
    if not lines:
        return None
    chosen = lines[-1] if prefer_last_line else lines[0]
    chosen = " ".join(chosen.split())
    if len(chosen) <= limit:
        return chosen
    return chosen[: limit - 3] + "..."


def _needs_review_reason(record):
    warnings = record.get("module_warnings") or []
    errors = record.get("module_errors") or []
    if warnings:
        return _compact_review_text(warnings[0]) or "A tool step was skipped or incomplete."
    if errors:
        return _compact_review_text(errors[0]) or "A tool step failed and needs checking."
    if record.get("program_error"):
        return "The Python step failed or produced incomplete output."
    if record.get("wolfram_error") or (record.get("wolfram_query") and not record.get("wolfram_output")):
        return "Wolfram did not return a usable answer."
    if not record.get("final_answer"):
        return "No clear final answer was extracted."
    if record.get("evaluation_status") == "evaluated" and record.get("is_correct") is False:
        mismatch_error = clean_display_text(record.get("evaluation_error"))
        if mismatch_error:
            return mismatch_error
        return "The final answer did not match the expected answer."
    return "One or more tool outputs disagreed or were incomplete."





def _needs_review_panel(record):
    if record.get("status") != "needs-review":
        return ""

    return f"""
    <section class="review-panel">
      <div class="review-panel__title">Needs Review</div>
      <p class="review-panel__reason">{_render_text_html(_needs_review_reason(record), preserve_breaks=False)}</p>
    </section>
    """


def _method_answer_candidates(record):
    options = record.get("options")
    problem = record.get("problem")
    candidates = []
    source_specs = [
        (
            "Solution Generator",
            {
                "solution": record.get("solution_generator_output") or record.get("final_generated_solution"),
                "program_output": record.get("program_output"),
                "wolfram_output": record.get("wolfram_output"),
                "options": options,
                "problem": problem,
            },
        ),
        ("Python", {"answer": record.get("program_output"), "options": options, "problem": problem}),
        ("Wolfram", {"answer": record.get("wolfram_output"), "options": options, "problem": problem}),
    ]

    for label, payload in source_specs:
        answer = resolve_final_answer_bundle(payload).get("display")
        if answer:
            candidates.append((label, answer))

    knowledge_text = record.get("knowledge_retrieval_output")
    if knowledge_text not in (None, ""):
        tagged_answer = extract_tagged_answer(knowledge_text or "")
        option_answer = infer_correct_option(knowledge_text or "")
        if tagged_answer:
            answer = resolve_final_answer_bundle({"final_answer": tagged_answer, "options": options, "problem": problem}).get("display")
        elif option_answer and options:
            answer = format_option_answer(options, option_answer)
        else:
            lines = [line.strip("- ").strip() for line in clean_display_text(knowledge_text or "").splitlines() if line.strip()]
            short_answer = lines[0] if len(lines) == 1 and len(lines[0]) <= 120 else None
            answer = resolve_final_answer_bundle({"final_answer": short_answer, "options": options, "problem": problem}).get("display")
        if answer:
            candidates.append(("Knowledge", answer))
    return candidates


def _method_source_for_answer(record, target_answer, fallback_label):
    if target_answer in (None, ""):
        return fallback_label, None

    for label, candidate_answer in _method_answer_candidates(record):
        if answers_match(record.get("dataset"), candidate_answer, target_answer, options=record.get("options")):
            return label, candidate_answer
    return fallback_label, target_answer


def _incorrect_evaluation_reason(record):
    mismatch_error = clean_display_text(record.get("evaluation_error"))
    if mismatch_error:
        final_answer = record.get("predicted_answer") or record.get("final_answer")
        final_source, _ = _method_source_for_answer(record, final_answer, "Final Answer")
        if final_source != "Final Answer":
            return f"{final_source}: {mismatch_error}"
        return mismatch_error
    final_answer = record.get("predicted_answer") or record.get("final_answer")
    gold_answer = record.get("gold_answer")
    final_source, _ = _method_source_for_answer(record, final_answer, "Final Answer")
    if final_source == "Final Answer":
        return "The final answer did not match the expected gold answer."
    return f"{final_source} produced the final answer, but it did not match the expected gold answer."


def _incorrect_evaluation_panel(record):
    if record.get("status") != "incorrect-evaluation":
        return ""

    return f"""
    <section class="review-panel review-panel--incorrect">
      <div class="review-panel__title">Incorrect Answer</div>
      <p class="review-panel__reason">{_render_text_html(_incorrect_evaluation_reason(record), preserve_breaks=False)}</p>
    </section>
    """


def _attention_panel(record):
    return _needs_review_panel(record) or _incorrect_evaluation_panel(record)


def _looks_like_inline_math_fragment(text):
    candidate = clean_display_text(text or "").strip()
    if not candidate or "\n" in candidate:
        return False
    if candidate.startswith("$") or candidate.endswith("$"):
        return False
    if re.fullmatch(r"[A-Za-z]", candidate):
        return True
    if re.fullmatch(r"[A-Z]{2,12}", candidate):
        return True
    if re.fullmatch(r"[A-Za-z][A-Za-z0-9_']{0,11}", candidate):
        return True
    if "(" in candidate and ")" in candidate and re.search(r"[A-Za-z0-9]", candidate):
        return True
    math_markers = (
        "\\frac",
        "\\sqrt",
        "\\boxed",
        "\\cdot",
        "\\times",
        "\\pi",
        "\\star",
        "\\ast",
        "\\infty",
        "\\cup",
        "\\cap",
        "\\pm",
        "\\neq",
        "\\leq",
        "\\geq",
        "^",
        "_",
        "{",
        "}",
        "=",
    )
    if any(marker in candidate for marker in math_markers):
        return True
    if re.search(r"\\[A-Za-z]+", candidate):
        return True
    if candidate in {"*", "+", "-", "=", "<", ">", "\\star", "\\ast", "\\times", "\\div", "\\pm", "\\neq", "\\leq", "\\geq"}:
        return True
    if re.search(r"[A-Za-z]\s*[\+\-\*/=]\s*[A-Za-z0-9]", candidate):
        return True
    if re.search(r"\d+\s*[\+\-\*/=]\s*\d+", candidate):
        return True
    return False


def _protect_currency_fragments(text):
    protected = []

    def replace_currency(match):
        protected.append(match.group(0))
        return f"__MATHSENSEI_CURRENCY_{len(protected) - 1}__"

    content = re.sub(r"(?<!\\)\$-?\d[\d,]*(?:\.\d+)?(?=(?:\s|$|[)\],.;:!?]))", replace_currency, str(text))
    return content, protected


def _restore_currency_fragments(text, protected):
    restored = str(text)
    for idx, original in enumerate(protected):
        restored = restored.replace(f"__MATHSENSEI_CURRENCY_{idx}__", original)
    return restored


def _convert_inline_math_delimiters(text):
    if text is None:
        return None

    content, protected_currency = _protect_currency_fragments(text)

    def replace_display(match):
        inner = match.group(1).strip()
        if not inner:
            return match.group(0)
        return f"\\[{inner}\\]"

    def replace_inline(match):
        inner = match.group(1).strip()
        if not inner or not _looks_like_inline_math_fragment(inner):
            return match.group(0)
        return f"\\({inner}\\)"

    content = re.sub(r"\$\$([\s\S]+?)\$\$", replace_display, content)
    content = re.sub(r"(?<!\\)\$([^\$\n]+?)\$", replace_inline, content)
    return _restore_currency_fragments(content, protected_currency)


def _auto_wrap_math_lines(text):
    if text is None:
        return None

    wrapped_lines = []
    for raw_line in str(text).splitlines():
        if not raw_line.strip():
            wrapped_lines.append(raw_line)
            continue

        if any(token in raw_line for token in ("\\(", "\\)", "\\[", "\\]", "$$")):
            wrapped_lines.append(raw_line)
            continue

        stripped = raw_line.strip()
        prefixed_match = re.match(r"^(\s*(?:[A-Z][\).:]\s+|[^:\n]{1,40}:\s+))(.+?)\s*$", raw_line)
        if prefixed_match and _looks_like_math(prefixed_match.group(2)):
            wrapped_lines.append(f"{prefixed_match.group(1)}\\({prefixed_match.group(2).strip()}\\)")
            continue

        sentence_match = re.match(r"^(\s*.*?\b(?:is|are|equals|equal to)\s+)(.+?)\s*$", raw_line, flags=re.IGNORECASE)
        if sentence_match and _looks_like_math(sentence_match.group(2)):
            wrapped_lines.append(f"{sentence_match.group(1)}\\({sentence_match.group(2).strip()}\\)")
            continue

        if _looks_like_math(stripped):
            leading = raw_line[: len(raw_line) - len(raw_line.lstrip())]
            trailing = raw_line[len(raw_line.rstrip()) :]
            wrapped_lines.append(f"{leading}\\({stripped}\\){trailing}")
            continue

        wrapped_lines.append(raw_line)

    return "\n".join(wrapped_lines)


def _looks_like_math(text):
    if not text:
        return False

    candidate = str(text).strip()
    if not candidate:
        return False

    if any(token in candidate for token in ("$", "\\(", "\\)", "\\[", "\\]")):
        return False

    math_markers = (
        "\\frac",
        "\\sqrt",
        "\\boxed",
        "\\cdot",
        "\\times",
        "\\pi",
        "\\infty",
        "\\cup",
        "\\cap",
        "\\pm",
        "\\neq",
        "\\leq",
        "\\geq",
        "^",
        "_",
        "{",
        "}",
        "=",
    )
    if any(marker in candidate for marker in math_markers):
        return True
    if re.search(r"\\[A-Za-z]+", candidate):
        return True

    # Short symbolic answers like [0,1), x+2, 48/95, A, or 3pi.
    return bool(re.fullmatch(r"[\[\]\(\)\{\}0-9A-Za-z,\.\\\-+/=*<>≤≥\s]+", candidate)) and len(candidate.split()) <= 4


def _strip_display_markdown(text):
    if text is None:
        return None

    cleaned = str(text)
    cleaned = re.sub(r"^\s*```[A-Za-z0-9_-]*\s*$", "", cleaned, flags=re.MULTILINE)
    cleaned = cleaned.replace("**", "")
    cleaned = cleaned.replace("__", "")
    cleaned = cleaned.replace("`", "")

    normalized_lines = []
    for raw_line in cleaned.splitlines():
        line = raw_line.rstrip()
        line = re.sub(r"^\s{0,3}#{1,6}\s*", "", line)
        line = re.sub(r"^\s*>\s?", "", line)
        line = re.sub(r"^\s*[-*]\s+", "", line)
        normalized_lines.append(line)

    cleaned = "\n".join(normalized_lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _looks_like_structured_literal_text(text):
    if text is None:
        return False

    candidate = str(text).strip()
    if not candidate:
        return False

    if candidate[0] not in "{[(":
        return False

    try:
        parsed = json.loads(candidate)
    except Exception:
        try:
            parsed = ast.literal_eval(candidate)
        except Exception:
            return False

    return isinstance(parsed, (dict, list, tuple))


def _strip_python_comments(text):
    try:
        tokens = tokenize.generate_tokens(io.StringIO(str(text)).readline)
        rebuilt = []
        previous_end = (1, 0)
        for token in tokens:
            if token.type == tokenize.COMMENT:
                previous_end = token.end
                continue
            if token.type == tokenize.ENDMARKER:
                break
            start_line, start_col = token.start
            end_line, end_col = token.end
            prev_line, prev_col = previous_end
            if start_line > prev_line:
                rebuilt.append("\n" * (start_line - prev_line))
                rebuilt.append(" " * start_col)
            elif start_col > prev_col:
                rebuilt.append(" " * (start_col - prev_col))
            rebuilt.append(token.string)
            previous_end = (end_line, end_col)
        return "".join(rebuilt)
    except Exception:
        return str(text)


def _clean_code_display(text):
    if text is None:
        return None

    cleaned = str(text)
    cleaned = re.sub(r"^\s*```[A-Za-z0-9_-]*\s*$", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"/\*[\s\S]*?\*/", "", cleaned)
    cleaned = _strip_python_comments(cleaned)

    normalized_lines = []
    for raw_line in cleaned.splitlines():
        line = raw_line.rstrip()
        line = re.sub(r"^\s{0,3}#{1,6}\s*", "", line)
        if re.match(r"^\s*(#|//|/\*|\*)", line):
            continue
        line = re.sub(r"^\s*python\s*:?\s*$", "", line, flags=re.IGNORECASE)
        normalized_lines.append(line)

    cleaned = "\n".join(normalized_lines)
    cleaned = re.sub(r"\n\s*\n+", "\n", cleaned)
    return cleaned.strip()


def _render_preformatted_html(text, strip_comments=False):
    if text is None:
        return ""

    cleaned = _clean_code_display(text) if strip_comments else _strip_display_markdown(text)
    return escape(str(cleaned))


def _render_plain_text_html(text, wrap_math=False, preserve_breaks=True, auto_wrap_math=True):
    if text is None:
        return ""

    display_text = _strip_display_markdown(text)
    if not _looks_like_structured_literal_text(display_text):
        display_text = _convert_inline_math_delimiters(display_text)
        if auto_wrap_math:
            display_text = _auto_wrap_math_lines(display_text)
    already_wrapped = any(token in str(display_text) for token in ("\\(", "\\)", "\\[", "\\]", "$$"))
    rendered = escape(str(display_text))
    if wrap_math and not already_wrapped and _looks_like_math(display_text):
        rendered = f"\\({rendered}\\)"
    if preserve_breaks:
        return rendered.replace("\n", "<br>")
    return rendered.replace("\n", " ")


def _render_text_html(text, wrap_math=False, preserve_breaks=True):
    if text is None:
        return ""

    if _looks_like_structured_literal_text(text):
        return _render_plain_text_html(text, wrap_math=wrap_math, preserve_breaks=preserve_breaks)

    parts = split_asy_blocks(text)
    if len(parts) == 1 and parts[0][0] == "text":
        return _render_plain_text_html(text, wrap_math=wrap_math, preserve_breaks=preserve_breaks)

    rendered_parts = []
    for part_type, value in parts:
        if part_type == "asy":
            rendered_parts.append(render_asy_block_html(value))
        else:
            chunk = _render_plain_text_html(value, wrap_math=wrap_math, preserve_breaks=preserve_breaks)
            if chunk:
                rendered_parts.append(chunk)
    return "".join(rendered_parts)


def _format_question_display_text(text):
    cleaned = _strip_display_markdown(clean_display_text(text) or "")
    if not cleaned:
        return ""
    collapsed = re.sub(r"[ \t]+", " ", cleaned).strip()

    key_match = re.search(r"(?i)\btranslation key:\s*", collapsed)
    if not key_match:
        return collapsed

    header = collapsed[: key_match.end()].strip()
    tail = collapsed[key_match.end() :].strip()
    if not tail:
        return header

    proposition_match = re.search(r"[\(\[\{]\s*[∃∀]", tail)
    definitions_text = tail
    proposition_text = ""
    if proposition_match:
        definitions_text = tail[: proposition_match.start()].strip(" ,;")
        proposition_text = tail[proposition_match.start() :].strip()

    definition_lines = []
    definition_matches = list(re.finditer(r"\b[A-Za-z][A-Za-z0-9]{0,5}\s*:\s*", definitions_text))
    if len(definition_matches) >= 2:
        for idx, match in enumerate(definition_matches):
            start = match.start()
            end = definition_matches[idx + 1].start() if idx + 1 < len(definition_matches) else len(definitions_text)
            chunk = definitions_text[start:end].strip(" ,;")
            if chunk:
                definition_lines.append(chunk)
    elif definitions_text:
        definition_lines.append(definitions_text)

    lines = [header]
    lines.extend(definition_lines)
    if proposition_text:
        lines.append(proposition_text)
    return "\n".join(line for line in lines if line)


def _render_question_text_html(text, preserve_breaks=True):
    if text is None:
        return ""

    if _looks_like_structured_literal_text(text):
        return _render_plain_text_html(text, preserve_breaks=preserve_breaks, auto_wrap_math=False)

    parts = split_asy_blocks(text)
    if len(parts) == 1 and parts[0][0] == "text":
        formatted = _format_question_display_text(text)
        return _render_plain_text_html(formatted, preserve_breaks=preserve_breaks, auto_wrap_math=False)

    rendered_parts = []
    for part_type, value in parts:
        if part_type == "asy":
            rendered_parts.append(render_asy_block_html(value))
        else:
            formatted = _format_question_display_text(value)
            chunk = _render_plain_text_html(formatted, preserve_breaks=preserve_breaks, auto_wrap_math=False)
            if chunk:
                rendered_parts.append(chunk)
    return "".join(rendered_parts)


def _canonicalize_math_answer_text(text):
    if text is None:
        return None

    candidate = _strip_display_markdown(clean_display_text(text) or "").strip()
    if not candidate:
        return candidate

    for start, end in (("\\(", "\\)"), ("\\[", "\\]"), ("$", "$")):
        if candidate.startswith(start) and candidate.endswith(end):
            candidate = candidate[len(start):-len(end)].strip()
            break

    collapsed = _normalize_math_text(candidate) if _looks_like_math(candidate) else candidate
    if collapsed is None:
        return candidate

    if parse_expr is None or sympy_latex is None or "\n" in candidate:
        return collapsed

    parse_candidate = _latex_to_sympy_expression(candidate) or candidate
    replacements = {
        "^": "**",
        "âˆ’": "-",
        "â€“": "-",
        "â€”": "-",
        "Ã—": "*",
        "Ã·": "/",
    }
    for old, new in replacements.items():
        parse_candidate = parse_candidate.replace(old, new)

    try:
        transformations = standard_transformations + tuple(
            transform
            for transform in (implicit_multiplication_application, convert_xor)
            if transform is not None
        )
        parsed = parse_expr(parse_candidate, transformations=transformations, evaluate=True)
        return sympy_latex(parsed)
    except Exception:
        return collapsed


def _render_answer_like_html(text, preserve_breaks=False):
    return _render_text_html(_canonicalize_math_answer_text(text), wrap_math=True, preserve_breaks=preserve_breaks)


def _display_problem_number(pid):
    if isinstance(pid, int):
        return pid + 1

    try:
        return int(pid) + 1
    except (TypeError, ValueError):
        return "?"


def _detail_block(title, value, code=False):
    if not value:
        return ""

    render_as_code = code or _looks_like_structured_literal_text(value)
    tag = "pre" if render_as_code else "div"
    class_name = "code-block" if render_as_code else "rich-text"
    return f"""
    <details class="detail-block">
      <summary>{escape(title)}</summary>
      <{tag} class="{class_name}">{_render_preformatted_html(value) if render_as_code else _render_text_html(value, preserve_breaks=True)}</{tag}>
    </details>
    """


def _module_step_block(step, index):
    items = step.get("items", [])
    meta_items = step.get("meta", [])
    show_item_labels = len(items) > 1
    items_html = []
    for item in items:
        render_as_code = bool(item.get("code")) or _looks_like_structured_literal_text(item["value"])
        tag = "pre" if render_as_code else "div"
        class_name = "code-block" if render_as_code else "rich-text"
        rendered_value = _render_preformatted_html(item["value"], strip_comments=bool(item.get("strip_comments"))) if render_as_code else _render_text_html(item["value"], preserve_breaks=True)
        items_html.append(
            f"""
            <div class="step-item">
              {'<div class="step-item__label">' + escape(item['label']) + '</div>' if show_item_labels else ''}
              <{tag} class="{class_name}">{rendered_value}</{tag}>
            </div>
            """
        )

    meta_html = ""
    if meta_items:
        meta_html = f"""
        <div class="module-meta-strip">
          {''.join(
              f'<div class="module-meta-pill"><span class="module-meta-pill__label">{escape(meta_item["label"])}</span><span class="module-meta-pill__value">{escape(str(meta_item["value"]))}</span></div>'
              for meta_item in meta_items
          )}
        </div>
        """

    if not items_html and not meta_html:
        return ""

    return f"""
    <details class="detail-block module-step">
      <summary><span class="step-index">Step {index + 1}</span> {escape(step['title'])}</summary>
      <div class="step-items">
        {meta_html}
        {''.join(items_html)}
      </div>
    </details>
    """


def _distinct_known_values(records, value_fn):
    values = []
    for record in records:
        value = value_fn(record)
        if value in (None, "", "Unknown"):
            continue
        values.append(str(value))
    return sorted(set(values))


def _problem_type_label(dataset):
    return "Problem Type"


def _report_context(records, dataset):
    datasets = _distinct_known_values(records, lambda record: record.get("dataset"))
    problem_types = _distinct_known_values(records, lambda record: record.get("problem_type"))
    program_states = _distinct_known_values(records, _program_status_value)
    statuses = _distinct_known_values(records, lambda record: record.get("status"))

    return {
        "dataset": dataset,
        "show_dataset": len(datasets) > 1,
        "show_problem_type": len(problem_types) > 1,
        "show_program_meta": len(program_states) > 1,
        "show_status_chip": len(statuses) > 1,
    }


def _program_status_value(record):
    if record.get("program_error"):
        return "Error"
    if record.get("generated_program"):
        return "OK"
    return "Not used"


def _distribution_values(records, value_fn, sort_key=None):
    counts = Counter()
    for record in records:
        value = value_fn(record)
        if value in (None, "", "Unknown"):
            continue
        counts[str(value)] += 1
    if sort_key is not None:
        return dict(sorted(counts.items(), key=lambda item: sort_key(item[0])))
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def _dataset_distribution_values(records):
    counts = {dataset: 0 for dataset in DATASET_DASHBOARD_ORDER}
    for record in records:
        dataset = clean_display_text(record.get("dataset"))
        if dataset in counts:
            counts[dataset] += 1
    return counts


def _dataset_distribution_sections(dataset, summary, records):
    if dataset == "MIXED":
        sections = [_distribution_rows("Datasets", _dataset_distribution_values(records))]
        sections.append(_maybe_distribution(_problem_type_label(dataset), _distribution_values(records, lambda record: record.get("problem_type"))))
        sections.append(_maybe_distribution("Levels", _distribution_values(records, lambda record: record.get("level"), sort_key=_level_sort_key)))
        return sections

    if dataset == "MMLU":
        return [_maybe_distribution("Problem Types", _distribution_values(records, lambda record: record.get("problem_type")))]

    if dataset == "MATH":
        sections = [_maybe_distribution("Problem Types", summary["types"])]
        sections.append(_maybe_distribution("Levels", summary["levels"]))
        return sections

    return []


def _dataset_filter_config(dataset, records):
    configs = []
    dataset_options = _sort_filter_values("dataset", {record["dataset"] for record in records if record.get("dataset") not in (None, "", "Unknown")})
    type_options = _sort_filter_values("type", {record["problem_type"] for record in records if record["problem_type"] != "Unknown"})
    level_options = _sort_filter_values("level", {record["level"] for record in records if record.get("level") not in (None, "", "Unknown")})

    if len(dataset_options) > 1:
        configs.append(
            {
                "id": "datasetFilter",
                "kind": "base",
                "label": "Dataset",
                "attr": "data-dataset",
                "options": _filter_options(dataset_options),
            }
        )

    if dataset == "MATH":
        configs.extend([
            {
                "id": "typeFilter",
                "kind": "base",
                "label": _problem_type_label(dataset),
                "attr": "data-type",
                "options": _filter_options(type_options),
            },
        ])
    elif dataset == "MMLU":
        configs.extend([
            {
                "id": "typeFilter",
                "kind": "base",
                "label": _problem_type_label(dataset),
                "attr": "data-type",
                "options": _filter_options(type_options),
            },
        ])
    else:
        configs.extend([
            {
                "id": "typeFilter",
                "kind": "base",
                "label": _problem_type_label(dataset),
                "attr": "data-type",
                "options": _filter_options(type_options),
            },
        ])

    configs.append(
        {
            "id": "levelFilter",
            "kind": "base",
            "label": "Level",
            "attr": "data-level",
            "options": _filter_options(level_options),
        }
    )

    for label in METHOD_SCOREBOARD_ORDER:
        options = _method_filter_options(records, label)
        if not options:
            continue
        configs.append(
            {
                "id": f"{_method_filter_slug(label)}Filter",
                "kind": "method",
                "label": label,
                "attr": _method_filter_attr_name(label),
                "options": options,
            }
        )

    return [
        config
        for config in configs
        if (config.get("kind") == "method" and len(config["options"]) > 0)
        or (config.get("kind") != "method" and len(config["options"]) > 1)
    ]


def _render_option_board(record):
    options = record.get("options") or []
    if not options:
        return ""

    option_cards = []
    for option in options:
        key = option.get("key", "")
        label = option.get("label", "")
        option_cards.append(
            f"""
            <div class="quiz-option">
              <span class="quiz-option__key">{escape(key)}</span>
              <span class="quiz-option__text">{_render_text_html(label, preserve_breaks=False)}</span>
            </div>
            """
        )

    return f"""
    <section class="quiz-board">
      <div class="quiz-board__header">
        <span class="quiz-board__label">Options</span>
      </div>
      <div class="quiz-grid">
        {''.join(option_cards)}
      </div>
    </section>
    """


def _display_gold_answer(record):
    return record.get("gold_answer_display") or resolve_gold_answer_bundle(record).get("display") or record.get("gold_answer")


def _flagged_method_labels(record):
    return [
        row.get("label")
        for row in (record.get("method_evaluations") or [])
        if row.get("status") in {"incorrect", "error", "incomplete"}
    ]


def _evaluation_label(record):
    if record.get("status") == "needs-review":
        return "Needs Review"
    overall_label = record.get("overall_evaluation_label")
    if overall_label:
        return overall_label
    if record.get("final_answer"):
        return "Not Evaluated"
    return "No Answer"


def _evaluation_explanation(record):
    if record.get("status") == "complete":
        flagged_methods = _flagged_method_labels(record)
        if flagged_methods:
            return (
                "The final answer matches the gold answer. "
                + "\nFlagged method checks: "
                + ", ".join(flagged_methods)
                + "."
            )
        return "The final answer matches the gold answer."
    if record.get("status") == "incorrect-evaluation":
        return _incorrect_evaluation_reason(record)
    if record.get("status") == "needs-review":
        return _needs_review_reason(record)
    if record.get("final_answer"):
        return "A final answer was extracted, but the report could not compare it to the gold answer automatically."
    return "No clear final answer was extracted from the generated work."


def _method_evaluation_board(record):
    rows = record.get("method_evaluations") or []
    if not rows:
        return ""

    return f"""
    <section class="method-board">
      <div class="method-board__label">Method Checks</div>
      <div class="method-board__list">
        {''.join(
            f'''
            <div class="method-check method-check--{row["status"]}">
              <div class="method-check__heading">
                <span class="method-check__symbol">{row["symbol"]}</span>
                <span class="method-check__name">{escape(row["label"])}</span>
                <span class="method-check__status">{escape(row["status_label"])}</span>
              </div>
              <div class="method-check__answer">{_render_text_html("No answer extracted", preserve_breaks=False) if (row.get("answer") or "") == "No answer extracted" else _render_answer_like_html(row.get("answer") or "No answer extracted")}</div>
            </div>
            '''
            for row in rows
        )}
      </div>
    </section>
    """


def _timing_summary_html(record):
    total_elapsed = _format_elapsed_seconds(record.get("question_elapsed_seconds"))
    module_names = []
    for collection in (
        _normalize_modules_list(record.get("modules_used")),
        list(_normalize_seconds_mapping(record.get("module_timings_seconds")).keys()),
        list(_normalize_text_mapping(record.get("module_backends")).keys()),
    ):
        for module_name in collection:
            normalized_name = _normalize_module_name(module_name)
            if normalized_name and normalized_name not in module_names:
                module_names.append(normalized_name)
    module_names = _order_modules(module_names)

    rows = []
    for module_name in module_names:
        backend = _module_backend_value(record, module_name)
        elapsed = _format_elapsed_seconds(_module_elapsed_seconds(record, module_name))
        if not backend and not elapsed:
            continue
        rows.append(
            f"""
            <div class="timing-row">
              <div class="timing-row__title">{escape(_module_title(module_name))}</div>
              <div class="timing-row__backend">{escape(backend or 'Backend unavailable')}</div>
              <div class="timing-row__elapsed">{escape(elapsed or '-')}</div>
            </div>
            """
        )

    if not total_elapsed and not rows:
        return ""

    total_html = ""
    if total_elapsed:
        total_html = f"""
        <div class="timing-total">
          <span class="timing-total__label">Total Question Time</span>
          <span class="timing-total__value">{escape(total_elapsed)}</span>
        </div>
        """

    return f"""
    <section class="timing-board">
      <div class="timing-board__header">
        <div class="timing-board__title">Runtime</div>
        {total_html}
      </div>
      {'<div class="timing-list">' + ''.join(rows) + '</div>' if rows else ''}
    </section>
    """


def _answer_reveal_block(record):
    final_answer = record.get("final_answer") or "Answer not extracted"
    gold_answer = _display_gold_answer(record) or "Gold answer not available"
    evaluation = _evaluation_label(record)
    explanation = _evaluation_explanation(record)
    attention_panel = _attention_panel(record)
    timing_summary = _timing_summary_html(record)
    method_board = _method_evaluation_board(record)

    evaluation_class = "answer-summary-card--neutral"
    if evaluation == "Correct":
        evaluation_class = "answer-summary-card--correct"
    elif evaluation == "Incorrect":
        evaluation_class = "answer-summary-card--incorrect"
    elif evaluation == "Needs Review":
        evaluation_class = "answer-summary-card--review"

    return f"""
    <details class="detail-block reveal-block answer-reveal">
      <summary>Show Answer &amp; Evaluation</summary>
      <div class="reveal-body answer-reveal__body">
        <div class="answer-summary-grid">
          <section class="answer-summary-card">
            <span class="answer-summary-card__label">Final Answer</span>
            <div class="answer-summary-card__value">{_render_answer_like_html(final_answer)}</div>
          </section>
          <section class="answer-summary-card">
            <span class="answer-summary-card__label">Gold Answer</span>
            <div class="answer-summary-card__value">{_render_answer_like_html(gold_answer)}</div>
          </section>
          <section class="answer-summary-card {evaluation_class}">
            <span class="answer-summary-card__label">Overall Evaluation</span>
            <div class="answer-summary-card__value">{escape(evaluation)}</div>
          </section>
        </div>
        <section class="answer-note">
          <div class="answer-note__label">Explanation</div>
          <p class="answer-note__text">{_render_text_html(explanation, preserve_breaks=False)}</p>
        </section>
        {method_board}
        {timing_summary}
        {attention_panel}
      </div>
    </details>
    """


def _workings_reveal_block(detail_blocks):
    if not detail_blocks:
        return ""

    return f"""
    <details class="detail-block reveal-block workings-reveal">
      <summary>Show Solutions &amp; Workings</summary>
      <div class="reveal-body">
        <div class="detail-stack detail-stack--nested">
          {''.join(detail_blocks)}
        </div>
      </div>
    </details>
    """


def _method_status_data_attributes(record):
    attributes = []
    for row in record.get("method_evaluations") or []:
        label = clean_display_text(row.get("label"))
        status = clean_display_text(row.get("status"))
        if not label or not status:
            continue
        attributes.append(f'{_method_filter_attr_name(label)}="{escape(status)}"')
    return " ".join(attributes)


def _record_card(record, context):
    option_board = _render_option_board(record)

    module_steps = record.get("module_steps") or []
    detail_blocks = [_module_step_block(step, index) for index, step in enumerate(module_steps)]
    detail_blocks.extend(
        [
            _detail_block("Ground Truth Solution", record.get("ground_truth_solution")),
        ]
    )
    answer_reveal = _answer_reveal_block(record)
    workings_reveal = _workings_reveal_block(detail_blocks)
    method_status_attrs = _method_status_data_attributes(record)
    if method_status_attrs:
        method_status_attrs = " " + method_status_attrs

    return f"""
    <article class="problem-card" data-status="{record['status']}" data-dataset="{escape(str(record.get('dataset') or 'Unknown'))}" data-type="{escape(record['problem_type'])}" data-level="{escape(record['level'])}"{method_status_attrs}>
      <section class="question-shell">
        <div class="problem-card__eyebrow">Problem {_display_problem_number(record.get('pid'))}</div>
        <div class="problem-card__title">{_render_question_text_html(record['problem'], preserve_breaks=True)}</div>
      </section>
      {option_board}
      <div class="reveal-stack">
        {answer_reveal}
        {workings_reveal}
      </div>
    </article>
    """


def _format_accuracy_value(value):
    if value is None:
        return "-"
    return f"{value * 100:.2f}%"


def _info_tip_html(text):
    return f'<span class="info-tip" tabindex="0">i<span class="info-tip__bubble">{escape(text)}</span></span>'


def _method_accuracy_scorecard_html(summary):
    rows = summary.get("method_accuracy") or []
    if not rows:
        return """
        <div class="scoreboard-empty">No method-level answers were available to score in this report.</div>
        """

    return "".join(
        f"""
        <div class="scoreboard-row">
          <div class="scoreboard-row__model">{escape(row['label'])}</div>
          <div class="scoreboard-row__metric">{_format_accuracy_value(row['accuracy'])}</div>
          <div class="scoreboard-row__meta">{row['correct']} correct / {row['evaluated']} evaluated / {row['flagged']} flagged</div>
        </div>
        """
        for row in rows
    )


def render_html_report(records, output_path, title, source_path=None, run_benchmark=None, aggregate_benchmark=None, skipped_rows=0):
    sorted_records = _sorted_records(records)
    summary = build_summary(sorted_records)
    dataset = infer_dataset(records=sorted_records, source_path=source_path, title=title)
    context = _report_context(sorted_records, dataset)
    cards_html = "".join(_record_card(record, context) for record in sorted_records)
    distribution_sections = [section for section in _dataset_distribution_sections(dataset, summary, sorted_records) if section]
    filter_config = _dataset_filter_config(dataset, sorted_records)
    filter_controls_html = "".join(
        f"""
        <div class="control">
          <label for="{escape(config['id'])}">{escape(config['label'])}</label>
          <select id="{escape(config['id'])}" data-attr="{escape(config['attr'])}">
            <option value="">All {escape(config['label']).lower()}</option>
            {''.join(f'<option value="{escape(str(option["value"]))}">{escape(str(option["label"]))}</option>' for option in config['options'])}
          </select>
        </div>
        """
        for config in filter_config
    )
    distribution_html = "".join(distribution_sections)
    status_filter_html = ""
    if context["show_status_chip"]:
        status_filter_html = """
        <div class="control">
          <label for="statusFilter">Status</label>
          <select id="statusFilter">
            <option value="">All statuses</option>
            <option value="complete">Complete</option>
            <option value="incorrect-evaluation">Incorrect Evaluation</option>
            <option value="needs-review">Needs Review</option>
          </select>
        </div>
        """
    filter_row_html = filter_controls_html + status_filter_html

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(title)}</title>
  <style>
    :root {{
      --bg: #07111f;
      --bg-2: #0c1728;
      --bg-3: #132239;
      --panel: rgba(13, 22, 36, 0.84);
      --panel-strong: rgba(10, 18, 30, 0.96);
      --line: rgba(148, 163, 184, 0.16);
      --line-strong: rgba(148, 163, 184, 0.28);
      --text: #edf4ff;
      --muted: #95a7be;
      --accent: #5eead4;
      --accent-2: #60a5fa;
      --accent-soft: rgba(94, 234, 212, 0.14);
      --warning: #fbbf24;
      --warning-soft: rgba(251, 191, 36, 0.14);
      --danger: #fb7185;
      --danger-soft: rgba(251, 113, 133, 0.16);
      --shadow: 0 28px 80px rgba(0, 0, 0, 0.38);
      --shadow-soft: 0 14px 40px rgba(0, 0, 0, 0.24);
      --radius: 26px;
      --radius-sm: 18px;
      --mono: "Cascadia Code", "JetBrains Mono", "Consolas", monospace;
      --sans: "Aptos", "Segoe UI Variable Display", "Segoe UI", sans-serif;
      --display: "Aptos Display", "Bahnschrift", "Trebuchet MS", sans-serif;
    }}

    * {{
      box-sizing: border-box;
    }}

    body {{
      margin: 0;
      font-family: var(--sans);
      color: var(--text);
      background:
        radial-gradient(circle at 12% 18%, rgba(96, 165, 250, 0.26), transparent 22%),
        radial-gradient(circle at 86% 10%, rgba(94, 234, 212, 0.22), transparent 24%),
        radial-gradient(circle at 50% 120%, rgba(244, 114, 182, 0.10), transparent 30%),
        linear-gradient(160deg, var(--bg) 0%, var(--bg-2) 48%, var(--bg-3) 100%);
      min-height: 100vh;
      position: relative;
      overflow-x: hidden;
    }}

    body::before {{
      content: "";
      position: fixed;
      inset: 0;
      background:
        linear-gradient(rgba(255,255,255,0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,255,255,0.03) 1px, transparent 1px);
      background-size: 64px 64px;
      mask-image: radial-gradient(circle at center, black 34%, transparent 88%);
      pointer-events: none;
      opacity: 0.26;
    }}

    body::after {{
      content: "";
      position: fixed;
      width: 42rem;
      height: 42rem;
      right: -14rem;
      top: -12rem;
      background: radial-gradient(circle, rgba(96, 165, 250, 0.16), transparent 64%);
      filter: blur(14px);
      pointer-events: none;
    }}

    .shell {{
      width: min(1220px, calc(100vw - 72px));
      margin: 32px auto 72px;
      position: relative;
      z-index: 1;
    }}

    .hero {{
      background:
        radial-gradient(circle at top right, rgba(94, 234, 212, 0.16), transparent 26%),
        linear-gradient(145deg, rgba(14, 25, 41, 0.96), rgba(8, 15, 26, 0.9));
      border: 1px solid rgba(255, 255, 255, 0.08);
      box-shadow: var(--shadow);
      border-radius: calc(var(--radius) + 4px);
      padding: 32px 36px;
      backdrop-filter: blur(18px) saturate(130%);
      position: relative;
      overflow: visible;
    }}

    .hero::before {{
      content: "";
      position: absolute;
      inset: 0;
      background: linear-gradient(120deg, rgba(96,165,250,0.10), transparent 24%, transparent 76%, rgba(94,234,212,0.08));
      pointer-events: none;
    }}

    .hero-heading {{
      display: flex;
      align-items: center;
      gap: 10px;
      flex-wrap: wrap;
      position: relative;
      z-index: 1;
      margin-bottom: 24px;
    }}

    .hero-heading__title,
    .hero-heading__subtitle {{
      margin: 0;
      font-family: var(--display);
      line-height: 1.16;
      letter-spacing: -0.04em;
    }}

    .hero-heading__title {{
      font-size: clamp(1.7rem, 2.6vw, 2.4rem);
    }}

    .hero-heading__subtitle {{
      font-size: clamp(0.98rem, 1.35vw, 1.08rem);
      white-space: nowrap;
    }}

    .hero-heading__subtitle {{
      color: var(--muted);
    }}

    .summary-grid,
    .panel-grid,
    .problem-grid {{
      display: grid;
      gap: 20px;
    }}

    .summary-grid {{
      grid-template-columns: repeat(12, minmax(0, 1fr));
      margin-top: 20px;
      align-items: stretch;
    }}

    .panel-grid {{
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      margin: 26px auto 0;
    }}

    .panel,
    .summary-card,
    .problem-card {{
      background: var(--panel);
      border: 1px solid rgba(255, 255, 255, 0.07);
      border-radius: var(--radius);
      box-shadow: var(--shadow-soft);
      backdrop-filter: blur(18px) saturate(120%);
    }}

    .summary-card {{
      padding: 20px;
      position: relative;
      overflow: visible;
      isolation: isolate;
      min-height: 164px;
      display: flex;
      flex-direction: column;
      justify-content: flex-start;
      grid-column: span 3;
    }}

    .summary-card--scoreboard {{
      grid-column: span 8;
      min-height: auto;
    }}

    .summary-card--benchmark {{
      grid-column: span 4;
      min-height: auto;
    }}

    .summary-card--benchmark .value {{
      font-size: clamp(1.15rem, 2vw, 1.6rem);
      line-height: 1.2;
    }}

    .summary-card::before {{
      content: "";
      position: absolute;
      inset: 0;
      background: linear-gradient(160deg, rgba(96,165,250,0.10), transparent 48%, rgba(94,234,212,0.05));
      z-index: -1;
    }}

    .summary-card .label {{
      color: var(--muted);
      font-size: 0.72rem;
      display: block;
      margin-bottom: 10px;
      letter-spacing: 0.14em;
      text-transform: uppercase;
    }}

    .label-with-info {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
    }}

    .info-tip {{
      position: relative;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      width: 18px;
      height: 18px;
      border-radius: 999px;
      border: 1px solid rgba(255,255,255,0.16);
      background: rgba(255,255,255,0.05);
      color: #d9e7fb;
      font-size: 0.68rem;
      font-weight: 700;
      cursor: help;
      text-transform: none;
      letter-spacing: normal;
      flex-shrink: 0;
    }}

    .info-tip__bubble {{
      position: absolute;
      left: 50%;
      bottom: calc(100% + 10px);
      transform: translateX(-50%);
      width: min(260px, 70vw);
      padding: 10px 12px;
      border-radius: 12px;
      border: 1px solid rgba(255,255,255,0.12);
      background: rgba(6, 13, 24, 0.96);
      color: #dbe7f6;
      font-size: 0.76rem;
      line-height: 1.45;
      text-transform: none;
      letter-spacing: normal;
      white-space: normal;
      box-shadow: 0 14px 34px rgba(0,0,0,0.34);
      opacity: 0;
      pointer-events: none;
      transition: opacity 160ms ease;
      z-index: 12;
    }}

    .info-tip:hover .info-tip__bubble,
    .info-tip:focus .info-tip__bubble {{
      opacity: 1;
    }}

    .summary-card .value {{
      font-family: var(--display);
      font-size: clamp(1.8rem, 3vw, 2.45rem);
      font-weight: 700;
      letter-spacing: -0.04em;
      line-height: 1;
      overflow-wrap: anywhere;
      background: linear-gradient(135deg, #f8fbff 0%, #7dd3fc 50%, #5eead4 100%);
      -webkit-background-clip: text;
      background-clip: text;
      color: transparent;
    }}

    .summary-card .caption {{
      color: var(--muted);
      margin-top: 8px;
      font-size: 0.88rem;
      line-height: 1.5;
    }}

    .scoreboard-list {{
      display: grid;
      gap: 10px;
      margin-top: 6px;
    }}

    .scoreboard-row {{
      display: grid;
      grid-template-columns: minmax(0, 1.45fr) auto auto;
      gap: 12px;
      align-items: center;
      padding: 12px 14px;
      border-radius: 16px;
      border: 1px solid rgba(255,255,255,0.07);
      background: rgba(255,255,255,0.035);
    }}

    .scoreboard-row__model {{
      font-size: 0.92rem;
      line-height: 1.35;
      color: #f4f8ff;
      font-weight: 600;
    }}

    .scoreboard-row__metric {{
      font-family: var(--display);
      font-size: 1.02rem;
      letter-spacing: -0.02em;
      color: #8ee7d6;
      white-space: nowrap;
    }}

    .scoreboard-row__meta {{
      color: var(--muted);
      font-size: 0.78rem;
      white-space: nowrap;
    }}

    .scoreboard-empty {{
      color: var(--muted);
      font-size: 0.9rem;
      line-height: 1.6;
      margin-top: 4px;
    }}

    .panel {{
      padding: 22px 24px;
    }}

    .panel h3 {{
      margin: 0 0 16px;
      font-size: 0.84rem;
      text-transform: uppercase;
      letter-spacing: 0.16em;
      color: var(--muted);
    }}

    .controls {{
      display: grid;
      gap: 12px;
      width: min(960px, 100%);
      margin: 28px auto 26px;
      padding: 16px 18px;
      background: rgba(9, 17, 28, 0.78);
      border: 1px solid rgba(255, 255, 255, 0.07);
      border-radius: 22px;
      box-shadow: var(--shadow-soft);
      position: sticky;
      top: 10px;
      z-index: 5;
      backdrop-filter: blur(22px) saturate(135%);
    }}

    .controls-head {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 12px;
      align-items: end;
    }}

    .control {{
      display: flex;
      flex-direction: column;
      gap: 6px;
      min-width: 0;
    }}

    .control--search {{
      width: 100%;
    }}

    .results-summary {{
      display: inline-flex;
      align-items: center;
      justify-content: flex-end;
      flex-wrap: wrap;
      gap: 7px;
      min-width: 0;
      padding: 7px 11px;
      border-radius: 999px;
      border: 1px solid rgba(255,255,255,0.07);
      background: rgba(7, 15, 26, 0.72);
      box-shadow: inset 0 1px 0 rgba(255,255,255,0.03);
      transition: border-color 180ms ease, background 180ms ease;
    }}

    .results-summary[data-active="true"] {{
      border-color: rgba(94, 234, 212, 0.22);
      background: rgba(10, 25, 28, 0.74);
    }}

    .results-summary__label {{
      color: var(--muted);
      font-size: 0.67rem;
      text-transform: uppercase;
      letter-spacing: 0.14em;
      line-height: 1;
    }}

    .results-summary__value {{
      display: flex;
      align-items: baseline;
      gap: 5px;
      font-family: var(--display);
      font-size: 0.95rem;
      font-weight: 700;
      line-height: 1;
      color: #f8fbff;
      letter-spacing: -0.01em;
    }}

    .results-summary__slash {{
      color: rgba(148, 163, 184, 0.82);
      font-size: 0.84rem;
      font-weight: 600;
    }}

    .results-summary__caption {{
      color: rgba(169, 198, 239, 0.82);
      font-size: 0.76rem;
      line-height: 1;
      white-space: nowrap;
    }}

    .filter-row {{
      display: flex;
      gap: 12px;
      flex-wrap: nowrap;
      align-items: stretch;
      overflow-x: auto;
      padding-bottom: 2px;
      scrollbar-width: thin;
      scrollbar-color: rgba(96,165,250,0.55) rgba(255,255,255,0.05);
    }}

    .filter-row .control {{
      flex: 1 1 0;
      min-width: 180px;
    }}

    .control label {{
      color: var(--muted);
      font-size: 0.76rem;
      text-transform: uppercase;
      letter-spacing: 0.16em;
      padding-left: 4px;
    }}

    .control input,
    .control select {{
      width: 100%;
      border-radius: 14px;
      border: 1px solid rgba(255,255,255,0.09);
      background:
        linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.015)),
        var(--panel-strong);
      padding: 11px 13px;
      font: inherit;
      color: var(--text);
      box-shadow: inset 0 1px 0 rgba(255,255,255,0.03);
      transition: border-color 180ms ease, box-shadow 180ms ease, transform 180ms ease;
    }}

    .control input:focus,
    .control select:focus {{
      outline: none;
      border-color: rgba(94, 234, 212, 0.58);
      box-shadow: 0 0 0 4px rgba(94, 234, 212, 0.12);
      transform: translateY(-1px);
    }}

    .problem-grid {{
      grid-template-columns: repeat(2, minmax(0, 1fr));
      align-items: start;
      width: 100%;
      margin: 0 auto;
    }}

    .problem-card {{
      padding: 24px 26px;
      position: relative;
      overflow: hidden;
      min-width: 0;
      display: flex;
      flex-direction: column;
      transition: transform 220ms ease, border-color 220ms ease, box-shadow 220ms ease;
      animation: card-rise 520ms ease both;
    }}

    .problem-card::before {{
      content: "";
      position: absolute;
      left: 0;
      top: 0;
      width: 100%;
      height: 3px;
      background: linear-gradient(90deg, transparent, rgba(96,165,250,0.55), rgba(94,234,212,0.8), transparent);
      opacity: 0.8;
    }}

    .problem-card:hover {{
      transform: translateY(-2px);
      border-color: rgba(94, 234, 212, 0.18);
      box-shadow: 0 18px 52px rgba(0,0,0,0.34);
    }}

    .problem-card__eyebrow {{
      margin-bottom: 14px;
      color: #8fa7c5;
      font-size: 0.74rem;
      letter-spacing: 0.18em;
      text-transform: uppercase;
    }}

    .status-chip {{
      border-radius: 999px;
      padding: 8px 12px;
      font-size: 0.78rem;
      border: 1px solid rgba(255,255,255,0.09);
      backdrop-filter: blur(14px);
    }}

    .status-chip--complete {{
      background: linear-gradient(135deg, rgba(94, 234, 212, 0.18), rgba(34, 197, 94, 0.08));
      color: var(--accent);
      border-color: rgba(94, 234, 212, 0.32);
    }}

    .status-chip--needs-review {{
      background: linear-gradient(135deg, rgba(251,113,133,0.16), rgba(244,114,182,0.08));
      color: var(--danger);
      border-color: rgba(251, 113, 133, 0.28);
    }}

    .question-shell {{
      margin: 0 0 22px;
      padding: 0;
      border: 0;
      background: none;
      box-shadow: none;
      max-height: none;
      overflow: visible;
      min-width: 0;
    }}

    .problem-card__title {{
      margin: 0;
      font-family: var(--display);
      font-size: clamp(1.02rem, 1.4vw, 1.14rem);
      line-height: 1.7;
      font-weight: 600;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      letter-spacing: -0.01em;
      color: #f8fbff;
    }}

    .asy-diagram {{
      margin: 14px 0 6px;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      gap: 8px;
    }}

    .asy-diagram__toolbar {{
      width: min(100%, 360px);
      display: flex;
      justify-content: flex-end;
    }}

    .asy-diagram__expand {{
      padding: 6px 10px;
      border-radius: 999px;
      border: 1px solid rgba(127, 167, 245, 0.18);
      background: rgba(17, 27, 46, 0.72);
      color: #bdd6fa;
      font-size: 0.73rem;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      cursor: pointer;
      transition: border-color 180ms ease, transform 180ms ease, color 180ms ease, background 180ms ease;
    }}

    .asy-diagram__expand:hover {{
      transform: translateY(-1px);
      border-color: rgba(96, 165, 250, 0.34);
      color: #eef5ff;
      background: rgba(22, 35, 58, 0.88);
    }}

    .asy-diagram__frame {{
      display: flex;
      justify-content: center;
      align-items: center;
      width: min(100%, 360px);
      max-width: 100%;
      margin-inline: auto;
      padding: 12px;
      overflow: hidden;
      border-radius: 18px;
      background: linear-gradient(180deg, rgba(9, 17, 31, 0.94), rgba(13, 22, 39, 0.98));
      border: 1px solid rgba(127, 167, 245, 0.16);
      box-shadow:
        inset 0 1px 0 rgba(255,255,255,0.04),
        0 10px 24px rgba(0,0,0,0.16);
    }}

    .asy-diagram__svg {{
      width: auto;
      max-width: min(100%, 336px);
      max-height: 220px;
      height: auto;
      display: block;
    }}

    .asy-overlay {{
      position: fixed;
      inset: 0;
      z-index: 70;
      display: none;
      align-items: center;
      justify-content: center;
      padding: 28px;
    }}

    .asy-overlay[data-open="true"] {{
      display: flex;
    }}

    .asy-overlay__backdrop {{
      position: absolute;
      inset: 0;
      background: rgba(4, 9, 18, 0.76);
      backdrop-filter: blur(14px);
    }}

    .asy-overlay__dialog {{
      position: relative;
      width: min(96vw, 1080px);
      padding: 18px 18px 16px;
      border-radius: 24px;
      border: 1px solid rgba(127, 167, 245, 0.18);
      background: linear-gradient(180deg, rgba(11, 19, 34, 0.97), rgba(8, 15, 28, 0.98));
      box-shadow:
        0 32px 72px rgba(0,0,0,0.42),
        inset 0 1px 0 rgba(255,255,255,0.04);
    }}

    .asy-overlay__close {{
      margin-left: auto;
      margin-bottom: 12px;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      padding: 7px 11px;
      border-radius: 999px;
      border: 1px solid rgba(127, 167, 245, 0.18);
      background: rgba(17, 27, 46, 0.76);
      color: #dce9ff;
      font-size: 0.74rem;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      cursor: pointer;
    }}

    .asy-overlay__content {{
      display: flex;
      justify-content: center;
      align-items: center;
      min-height: min(56vh, 540px);
    }}

    .asy-overlay__content .asy-diagram__svg {{
      width: min(92vw, 960px);
      max-width: 100%;
      max-height: 78vh;
      height: auto;
    }}

    .asy-fallback {{
      margin: 14px 0 6px;
      border-radius: 16px;
      border: 1px solid rgba(251, 191, 36, 0.22);
      background: rgba(255, 255, 255, 0.03);
    }}

    .asy-fallback summary {{
      padding: 12px 14px;
      cursor: pointer;
      color: #fde68a;
      font-size: 0.8rem;
      letter-spacing: 0.12em;
      text-transform: uppercase;
    }}

    .quiz-board {{
      margin: 0 0 18px;
      padding: 0;
      border-radius: 0;
      background:
        none;
      border: 0;
      box-shadow: none;
    }}

    .quiz-board__header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
      margin-bottom: 12px;
      flex-wrap: wrap;
    }}

    .quiz-board__label {{
      font-size: 0.72rem;
      color: #a9c6ef;
      letter-spacing: 0.18em;
      text-transform: uppercase;
    }}

    .quiz-legend {{
      color: #7fe2a1;
      font-size: 0.88rem;
    }}

    .quiz-grid {{
      display: grid;
      grid-template-columns: 1fr;
      gap: 10px;
    }}

    .quiz-option {{
      position: relative;
      min-height: 62px;
      padding: 16px 18px;
      display: flex;
      align-items: flex-start;
      gap: 14px;
      color: #edf4ff;
      background:
        linear-gradient(180deg, rgba(19, 31, 54, 0.96), rgba(15, 25, 46, 0.98));
      border: 1px solid rgba(127, 167, 245, 0.16);
      border-radius: 20px;
      box-shadow:
        inset 0 1px 0 rgba(255,255,255,0.05),
        0 10px 24px rgba(0,0,0,0.16);
      overflow: hidden;
    }}

    .quiz-option::before {{
      content: "";
      position: absolute;
      inset: 0 auto 0 0;
      width: 4px;
      background: rgba(96, 165, 250, 0.55);
      pointer-events: none;
    }}

    .quiz-option--correct {{
      background:
        linear-gradient(180deg, rgba(12, 74, 53, 0.96), rgba(15, 92, 62, 0.98));
      border-color: rgba(110, 231, 183, 0.44);
      box-shadow:
        inset 0 1px 0 rgba(255,255,255,0.08),
        0 12px 30px rgba(16, 185, 129, 0.12);
    }}

    .quiz-option--correct::before {{
      background: rgba(110, 231, 183, 0.9);
    }}

    .quiz-option__key {{
      position: relative;
      z-index: 1;
      min-width: 32px;
      height: 32px;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      border-radius: 999px;
      background: rgba(255,255,255,0.08);
      border: 1px solid rgba(255,255,255,0.14);
      color: #f8fbff;
      font-weight: 700;
      letter-spacing: 0.06em;
      flex-shrink: 0;
    }}

    .quiz-option__text {{
      position: relative;
      z-index: 1;
      line-height: 1.55;
      font-size: 0.96rem;
      padding-top: 1px;
    }}

    .reveal-stack {{
      display: grid;
      gap: 12px;
    }}

    .reveal-block {{
      border-radius: 20px;
      background: rgba(255, 255, 255, 0.028);
    }}

    .reveal-block summary {{
      padding: 15px 18px;
      font-size: 0.92rem;
      letter-spacing: -0.01em;
    }}

    .reveal-block summary::after {{
      content: "+";
      float: right;
      color: #9bc2f5;
      font-size: 1rem;
      line-height: 1;
    }}

    .reveal-block[open] summary::after {{
      content: "-";
    }}

    .reveal-body {{
      padding: 0 18px 18px;
      border-top: 1px solid rgba(255,255,255,0.05);
    }}

    .answer-reveal__body {{
      display: grid;
      gap: 14px;
    }}

    .answer-summary-grid {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 12px;
      margin-top: 16px;
    }}

    .answer-summary-card {{
      padding: 14px 15px;
      border-radius: 18px;
      border: 1px solid rgba(255,255,255,0.07);
      background: linear-gradient(180deg, rgba(255,255,255,0.04), rgba(255,255,255,0.02));
      min-width: 0;
    }}

    .answer-summary-card--correct {{
      border-color: rgba(94, 234, 212, 0.3);
      background: linear-gradient(180deg, rgba(94, 234, 212, 0.10), rgba(255,255,255,0.02));
    }}

    .answer-summary-card--incorrect {{
      border-color: rgba(251, 191, 36, 0.34);
      background: linear-gradient(180deg, rgba(251, 191, 36, 0.10), rgba(255,255,255,0.02));
    }}

    .answer-summary-card--review {{
      border-color: rgba(251, 113, 133, 0.30);
      background: linear-gradient(180deg, rgba(251, 113, 133, 0.10), rgba(255,255,255,0.02));
    }}

    .answer-summary-card__label {{
      display: block;
      color: var(--muted);
      font-size: 0.72rem;
      text-transform: uppercase;
      letter-spacing: 0.16em;
      margin-bottom: 10px;
    }}

    .answer-summary-card__value {{
      font-size: 0.98rem;
      line-height: 1.55;
      word-break: break-word;
      overflow-wrap: anywhere;
      color: #f8fbff;
    }}

    .answer-meta-strip {{
      display: inline-flex;
      align-items: center;
      gap: 10px;
      flex-wrap: wrap;
      padding: 8px 12px;
      border-radius: 999px;
      border: 1px solid rgba(255,255,255,0.07);
      background: rgba(255,255,255,0.03);
      width: fit-content;
      max-width: 100%;
    }}

    .answer-meta-strip__label {{
      color: var(--muted);
      font-size: 0.7rem;
      text-transform: uppercase;
      letter-spacing: 0.16em;
    }}

    .answer-meta-strip__value {{
      font-size: 0.84rem;
      color: #dbe7f6;
    }}

    .answer-note {{
      padding: 14px 15px;
      border-radius: 18px;
      border: 1px solid rgba(255,255,255,0.06);
      background: rgba(255,255,255,0.03);
    }}

    .answer-note__label {{
      color: var(--muted);
      font-size: 0.72rem;
      text-transform: uppercase;
      letter-spacing: 0.16em;
      margin-bottom: 10px;
    }}

    .answer-note__text {{
      margin: 0;
      font-size: 0.94rem;
      line-height: 1.6;
      color: #edf4ff;
    }}

    .method-board {{
      display: grid;
      gap: 12px;
      padding: 14px 15px;
      border-radius: 18px;
      border: 1px solid rgba(255,255,255,0.07);
      background: rgba(255,255,255,0.03);
    }}

    .method-board__label {{
      color: var(--muted);
      font-size: 0.72rem;
      text-transform: uppercase;
      letter-spacing: 0.16em;
    }}

    .method-board__list {{
      display: grid;
      gap: 10px;
    }}

    .method-check {{
      display: grid;
      gap: 8px;
      padding: 12px 13px;
      border-radius: 16px;
      border: 1px solid rgba(255,255,255,0.06);
      background: rgba(255,255,255,0.025);
    }}

    .method-check--correct {{
      border-color: rgba(94, 234, 212, 0.28);
      background: linear-gradient(180deg, rgba(94, 234, 212, 0.09), rgba(255,255,255,0.02));
    }}

    .method-check--incorrect {{
      border-color: rgba(251, 191, 36, 0.28);
      background: linear-gradient(180deg, rgba(251, 191, 36, 0.09), rgba(255,255,255,0.02));
    }}

    .method-check--error {{
      border-color: rgba(251, 113, 133, 0.30);
      background: linear-gradient(180deg, rgba(251, 113, 133, 0.10), rgba(255,255,255,0.02));
    }}

    .method-check--incomplete {{
      border-color: rgba(148, 163, 184, 0.20);
      background: linear-gradient(180deg, rgba(148, 163, 184, 0.06), rgba(255,255,255,0.02));
    }}

    .method-check__heading {{
      display: flex;
      align-items: center;
      gap: 10px;
      flex-wrap: wrap;
    }}

    .method-check__symbol {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      width: 22px;
      height: 22px;
      border-radius: 999px;
      background: rgba(255,255,255,0.08);
      color: #f8fbff;
      font-size: 0.84rem;
      line-height: 1;
      flex-shrink: 0;
    }}

    .method-check__name {{
      color: #f4f8ff;
      font-size: 0.9rem;
      font-weight: 600;
    }}

    .method-check__status {{
      color: var(--muted);
      font-size: 0.78rem;
      text-transform: uppercase;
      letter-spacing: 0.12em;
    }}

    .method-check__answer {{
      color: #edf4ff;
      font-size: 0.92rem;
      line-height: 1.55;
      overflow-wrap: anywhere;
    }}

    .timing-board {{
      display: grid;
      gap: 12px;
      padding: 14px 15px;
      border-radius: 18px;
      border: 1px solid rgba(255,255,255,0.07);
      background: linear-gradient(180deg, rgba(96, 165, 250, 0.07), rgba(255,255,255,0.02));
    }}

    .timing-board__header {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      flex-wrap: wrap;
    }}

    .timing-board__title {{
      color: var(--muted);
      font-size: 0.72rem;
      text-transform: uppercase;
      letter-spacing: 0.16em;
    }}

    .timing-total {{
      display: inline-flex;
      align-items: center;
      gap: 10px;
      flex-wrap: wrap;
      padding: 8px 12px;
      border-radius: 999px;
      border: 1px solid rgba(96, 165, 250, 0.22);
      background: rgba(96, 165, 250, 0.10);
      max-width: 100%;
    }}

    .timing-total__label {{
      color: #b7cbe5;
      font-size: 0.7rem;
      text-transform: uppercase;
      letter-spacing: 0.14em;
    }}

    .timing-total__value {{
      font-family: var(--mono);
      font-size: 0.84rem;
      color: #eff6ff;
    }}

    .timing-list {{
      display: grid;
      gap: 10px;
    }}

    .timing-row {{
      display: grid;
      grid-template-columns: minmax(0, 1.15fr) minmax(0, 1fr) auto;
      gap: 12px;
      align-items: center;
      padding: 11px 12px;
      border-radius: 14px;
      border: 1px solid rgba(255,255,255,0.06);
      background: rgba(255,255,255,0.03);
    }}

    .timing-row__title {{
      color: #f4f8ff;
      font-size: 0.9rem;
      font-weight: 600;
    }}

    .timing-row__backend {{
      color: var(--muted);
      font-size: 0.8rem;
      line-height: 1.45;
      overflow-wrap: anywhere;
    }}

    .timing-row__elapsed {{
      font-family: var(--mono);
      font-size: 0.82rem;
      color: #dbeafe;
      white-space: nowrap;
    }}

    .detail-stack {{
      display: grid;
      gap: 10px;
      margin-top: 12px;
      min-width: 0;
    }}

    .detail-stack--nested {{
      margin-top: 16px;
    }}

    .detail-block {{
      border: 1px solid rgba(255,255,255,0.08);
      border-radius: 18px;
      background: rgba(255, 255, 255, 0.03);
      overflow: hidden;
      min-width: 0;
      transition: border-color 180ms ease, background 180ms ease;
    }}

    .detail-block[open] {{
      border-color: rgba(96, 165, 250, 0.22);
      background: rgba(255,255,255,0.04);
    }}

    .detail-block summary {{
      list-style: none;
      cursor: pointer;
      padding: 12px 14px;
      font-weight: 600;
      font-size: 0.88rem;
      background: rgba(255, 255, 255, 0.02);
      color: #eff6ff;
      transition: background 180ms ease;
    }}

    .detail-block:hover summary {{
      background: rgba(255,255,255,0.04);
    }}

    .detail-block summary::-webkit-details-marker {{
      display: none;
    }}

    .step-index {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-width: 56px;
      margin-right: 8px;
      padding: 4px 8px;
      border-radius: 999px;
      border: 1px solid rgba(96, 165, 250, 0.24);
      background: rgba(96, 165, 250, 0.12);
      color: #bfe0ff;
      font-size: 0.68rem;
      letter-spacing: 0.12em;
      text-transform: uppercase;
    }}

    .step-items {{
      display: grid;
      gap: 10px;
      padding: 12px 14px 14px;
      border-top: 1px solid rgba(255,255,255,0.04);
      min-width: 0;
    }}

    .module-meta-strip {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-bottom: 2px;
    }}

    .module-meta-pill {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      flex-wrap: wrap;
      padding: 7px 10px;
      border-radius: 999px;
      border: 1px solid rgba(255,255,255,0.08);
      background: rgba(255,255,255,0.035);
      max-width: 100%;
    }}

    .module-meta-pill__label {{
      color: var(--muted);
      font-size: 0.68rem;
      text-transform: uppercase;
      letter-spacing: 0.14em;
    }}

    .module-meta-pill__value {{
      color: #edf4ff;
      font-size: 0.8rem;
      line-height: 1.35;
      overflow-wrap: anywhere;
    }}

    .step-item {{
      display: grid;
      gap: 8px;
      min-width: 0;
    }}

    .step-item__label {{
      color: var(--muted);
      font-size: 0.74rem;
      text-transform: uppercase;
      letter-spacing: 0.14em;
      padding-left: 2px;
    }}

    .rich-text,
    .code-block {{
      margin: 0;
      padding: 12px;
      line-height: 1.45;
      white-space: pre-wrap;
      word-break: break-word;
      overflow-wrap: anywhere;
      max-width: 100%;
      max-height: 15rem;
      overflow: auto;
      min-width: 0;
      font-size: 0.87rem;
    }}

    .code-block {{
      font-family: var(--mono);
      font-size: 0.81rem;
      background: rgba(4, 10, 18, 0.92);
      color: #d8e6ff;
      border-top: 1px solid rgba(255,255,255,0.04);
      white-space: pre-wrap;
    }}

    .review-panel {{
      margin: 0;
      padding: 13px 14px;
      border-radius: 16px;
      border: 1px solid rgba(251, 113, 133, 0.28);
      background: linear-gradient(180deg, rgba(251, 113, 133, 0.08), rgba(255, 255, 255, 0.03));
    }}

    .review-panel__title {{
      font-size: 0.76rem;
      text-transform: uppercase;
      letter-spacing: 0.16em;
      color: #fecdd3;
      margin-bottom: 8px;
    }}

    .review-panel__reason {{
      margin: 0;
      font-size: 0.88rem;
      line-height: 1.4;
      color: #fce7f3;
    }}

    .review-panel--incorrect {{
      border-color: rgba(251, 191, 36, 0.28);
      background: linear-gradient(180deg, rgba(251, 191, 36, 0.08), rgba(255, 255, 255, 0.03));
    }}

    .review-panel--incorrect .review-panel__title {{
      color: #fde68a;
    }}

    .review-panel--incorrect .review-panel__reason {{
      color: #fef3c7;
    }}

    .review-compare {{
      display: grid;
      gap: 8px;
      margin-top: 10px;
    }}

    .review-compare__item {{
      display: grid;
      gap: 4px;
      padding: 8px 10px;
      border-radius: 12px;
      background: rgba(255, 255, 255, 0.035);
      border: 1px solid rgba(255, 255, 255, 0.06);
      min-width: 0;
    }}

    .review-compare__label {{
      color: var(--muted);
      font-size: 0.72rem;
      text-transform: uppercase;
      letter-spacing: 0.14em;
    }}

    .review-compare__value {{
      font-size: 0.84rem;
      line-height: 1.35;
      overflow-wrap: anywhere;
      max-height: 5.5rem;
      overflow: auto;
    }}

    .question-shell,
    .answer-card,
    .rich-text,
    .code-block {{
      scrollbar-width: thin;
      scrollbar-color: rgba(96,165,250,0.55) rgba(255,255,255,0.05);
    }}

    .question-shell::-webkit-scrollbar,
    .answer-card::-webkit-scrollbar,
    .rich-text::-webkit-scrollbar,
    .code-block::-webkit-scrollbar {{
      width: 10px;
      height: 10px;
    }}

    .question-shell::-webkit-scrollbar-thumb,
    .answer-card::-webkit-scrollbar-thumb,
    .rich-text::-webkit-scrollbar-thumb,
    .code-block::-webkit-scrollbar-thumb {{
      background: rgba(96, 165, 250, 0.48);
      border-radius: 999px;
      border: 2px solid rgba(8, 15, 26, 0.65);
    }}

    .question-shell::-webkit-scrollbar-track,
    .answer-card::-webkit-scrollbar-track,
    .rich-text::-webkit-scrollbar-track,
    .code-block::-webkit-scrollbar-track {{
      background: rgba(255,255,255,0.04);
      border-radius: 999px;
    }}

    .distribution-list {{
      display: grid;
      gap: 12px;
    }}

    .distribution-meta {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 7px;
      font-size: 0.94rem;
    }}

    .distribution-bar {{
      height: 10px;
      background: rgba(255,255,255,0.05);
      border-radius: 999px;
      overflow: hidden;
    }}

    .distribution-bar span {{
      display: block;
      height: 100%;
      border-radius: 999px;
      background: linear-gradient(90deg, #60a5fa, #5eead4);
      box-shadow: 0 0 18px rgba(94, 234, 212, 0.28);
    }}

    .muted,
    .empty-state {{
      color: var(--muted);
      line-height: 1.6;
    }}

    .problem-card__title mjx-container,
    .answer-value mjx-container,
    .rich-text mjx-container {{
      max-width: 100%;
      overflow-x: auto;
      overflow-y: hidden;
      padding-bottom: 2px;
    }}

    .problem-card__title mjx-container[display="true"],
    .answer-value mjx-container[display="true"],
    .rich-text mjx-container[display="true"] {{
      display: block;
    }}

    .empty-state {{
      display: none;
      padding: 28px;
      text-align: center;
      border: 1px dashed rgba(255,255,255,0.12);
      border-radius: var(--radius);
      background: rgba(255, 255, 255, 0.03);
      margin-top: 18px;
    }}

    @media (max-width: 1520px) {{
    }}

    @media (max-width: 1040px) {{
      .summary-grid {{
        grid-template-columns: repeat(8, minmax(0, 1fr));
      }}

      .summary-card {{
        grid-column: span 4;
      }}

      .summary-card--scoreboard,
      .summary-card--benchmark {{
        grid-column: span 8;
      }}
    }}

    @media (max-width: 1180px) {{
      .panel-grid {{
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }}
    }}

    @media (max-width: 820px) {{
      .summary-grid,
      .panel-grid,
      .problem-grid {{
        grid-template-columns: 1fr;
      }}

      .quiz-grid {{
        grid-template-columns: 1fr;
      }}

      .answer-summary-grid {{
        grid-template-columns: 1fr;
      }}

      .timing-row {{
        grid-template-columns: 1fr;
      }}

      .timing-row__elapsed {{
        white-space: normal;
      }}

      .summary-card--scoreboard {{
        grid-column: span 1;
      }}

      .summary-card--benchmark {{
        grid-column: span 1;
      }}

      .scoreboard-row {{
        grid-template-columns: 1fr;
        align-items: start;
      }}

      .scoreboard-row__meta,
      .scoreboard-row__metric {{
        white-space: normal;
      }}

      .shell {{
        width: min(100vw - 20px, 1380px);
        margin: 14px auto 36px;
      }}

      .hero-heading__title,
      .hero-heading__subtitle {{
        white-space: normal;
      }}

      .hero,
      .problem-card,
      .panel,
      .summary-card {{
        border-radius: 20px;
      }}

      .hero {{
        padding: 24px 22px;
      }}

      .problem-card {{
        padding: 20px 18px;
      }}

      .controls {{
        width: 100%;
        padding: 14px;
      }}

      .controls-head {{
        grid-template-columns: 1fr;
      }}

      .results-summary {{
        min-width: 0;
      }}

      .asy-diagram__frame {{
        width: min(100%, 308px);
      }}

      .asy-diagram__svg {{
        max-width: min(100%, 284px);
        max-height: 196px;
      }}

      .asy-diagram__toolbar {{
        width: min(100%, 308px);
      }}

      .asy-overlay {{
        padding: 14px;
      }}

      .asy-overlay__dialog {{
        width: min(100vw - 20px, 960px);
        padding: 14px;
      }}

      .asy-overlay__content {{
        min-height: min(48vh, 420px);
      }}
    }}

    @keyframes card-rise {{
      from {{
        opacity: 0;
        transform: translateY(18px);
      }}
      to {{
        opacity: 1;
        transform: translateY(0);
      }}
    }}
  </style>
</head>
<body>
  <main class="shell">
    <section class="hero">
      <div class="hero-heading">
        <h1 class="hero-heading__title">{escape(title)}</h1>
      </div>
      <div class="summary-grid">
        <div class="summary-card">
          <span class="label">Complete</span>
          <div class="value">{summary['complete_display']}</div>
          <div class="caption">Final answer matched the gold answer</div>
        </div>
        <div class="summary-card">
          <span class="label">Incorrect Evaluation</span>
          <div class="value">{summary['incorrect_evaluation']}</div>
          <div class="caption">Final answer did not match the gold answer</div>
        </div>
        <div class="summary-card">
          <span class="label">Needs Review</span>
          <div class="value">{summary['needs_review']}</div>
          <div class="caption">No reliable final evaluation could be produced</div>
        </div>
        <div class="summary-card summary-card--scoreboard">
          <span class="label label-with-info">Accuracy Scorecard {_info_tip_html("Each row scores one method by comparing its resolved answer value against the corresponding ground-truth answer using the same dataset-specific evaluator used elsewhere in the report.")}</span>
          <div class="scoreboard-list">
            {_method_accuracy_scorecard_html(summary)}
          </div>
        </div>
        <div class="summary-card summary-card--benchmark">
          <span class="label label-with-info">Benchmark {_info_tip_html("Benchmark identifies the run by its configured model and dataset.")}</span>
          <div class="value">{escape(run_benchmark['model']) if run_benchmark else 'N/A'}</div>
          <div class="caption">Dataset: {escape((run_benchmark or {}).get('dataset') or (aggregate_benchmark or {}).get('dataset') or dataset)}</div>
        </div>
      </div>
    </section>

    {'<section class="panel-grid">' + distribution_html + '</section>' if distribution_html else ''}

    <section class="controls">
      <div class="controls-head">
        <div class="control control--search">
          <label for="searchBox">Search</label>
          <input id="searchBox" type="search" placeholder="Search problems, answers, or solutions">
        </div>
        <div id="resultsSummary" class="results-summary" data-active="false" aria-live="polite">
          <div class="results-summary__label">Visible</div>
          <div class="results-summary__value">
            <span id="resultsVisibleCount">{len(sorted_records)}</span>
            <span class="results-summary__slash">/</span>
            <span id="resultsTotalCount">{len(sorted_records)}</span>
          </div>
          <div id="resultsCaption" class="results-summary__caption">All problems shown</div>
        </div>
      </div>
      {'<div class="filter-row">' + filter_row_html + '</div>' if filter_row_html else ''}
    </section>

    <section id="problemGrid" class="problem-grid">
      {cards_html}
    </section>
    <div id="emptyState" class="empty-state">No problems match the current filters.</div>
    <div id="asyOverlay" class="asy-overlay" hidden>
      <div class="asy-overlay__backdrop" data-close-asy="true"></div>
      <div class="asy-overlay__dialog" role="dialog" aria-modal="true" aria-label="Expanded diagram view">
        <button id="asyOverlayClose" type="button" class="asy-overlay__close">Close</button>
        <div id="asyOverlayContent" class="asy-overlay__content"></div>
      </div>
    </div>
  </main>

  <script>
    window.MathJax = {{
      tex: {{
        inlineMath: [['\\\\(', '\\\\)']],
        displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']],
        processEscapes: true
      }},
      chtml: {{
        displayOverflow: 'overflow'
      }},
      options: {{
        skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code']
      }}
    }};
  </script>
  <script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>
  <script>
    const searchBox = document.getElementById("searchBox");
    const statusFilter = document.getElementById("statusFilter");
    const datasetFilters = Array.from(document.querySelectorAll(".controls select[data-attr]"));
    const cards = Array.from(document.querySelectorAll(".problem-card"));
    const emptyState = document.getElementById("emptyState");
    const resultsSummary = document.getElementById("resultsSummary");
    const resultsVisibleCount = document.getElementById("resultsVisibleCount");
    const resultsTotalCount = document.getElementById("resultsTotalCount");
    const resultsCaption = document.getElementById("resultsCaption");
    const detailsBlocks = Array.from(document.querySelectorAll(".detail-block"));
    const totalCardCount = cards.length;
    const asyOverlay = document.getElementById("asyOverlay");
    const asyOverlayClose = document.getElementById("asyOverlayClose");
    const asyOverlayContent = document.getElementById("asyOverlayContent");
    const asyExpandButtons = Array.from(document.querySelectorAll(".asy-diagram__expand"));

    function applyFilters() {{
      const query = searchBox.value.trim().toLowerCase();
      const selectedStatus = statusFilter ? statusFilter.value : "";
      const activeDatasetFilterCount = datasetFilters.filter((filter) => filter.value).length;
      let visibleCount = 0;

      cards.forEach((card) => {{
        const matchesQuery = !query || card.textContent.toLowerCase().includes(query);
        const matchesStatus = !selectedStatus || card.dataset.status === selectedStatus;
        const matchesDatasetFilters = datasetFilters.every((filter) => {{
          const attr = filter.dataset.attr;
          const value = filter.value;
          if (!value) return true;
          const cardValue = card.getAttribute(attr) || "";
          if (value === "flagged") {{
            return cardValue && cardValue !== "correct";
          }}
          return cardValue === value;
        }});
        const visible = matchesQuery && matchesStatus && matchesDatasetFilters;
        card.style.display = visible ? "" : "none";
        if (visible) visibleCount += 1;
      }});

      emptyState.style.display = visibleCount === 0 ? "block" : "none";
      if (resultsVisibleCount) resultsVisibleCount.textContent = String(visibleCount);
      if (resultsTotalCount) resultsTotalCount.textContent = String(totalCardCount);
      if (resultsSummary) {{
        const activeFilterCount = (query ? 1 : 0) + (selectedStatus ? 1 : 0) + activeDatasetFilterCount;
        resultsSummary.dataset.active = activeFilterCount > 0 ? "true" : "false";
        if (resultsCaption) {{
          if (activeFilterCount > 0) {{
            resultsCaption.textContent = `${{activeFilterCount}} filter${{activeFilterCount === 1 ? "" : "s"}} active`;
          }} else {{
            resultsCaption.textContent = "All problems shown";
          }}
        }}
      }}
    }}

    [searchBox, statusFilter, ...datasetFilters].filter(Boolean).forEach((element) => {{
      element.addEventListener("input", applyFilters);
      element.addEventListener("change", applyFilters);
    }});

    applyFilters();

    window.addEventListener("load", () => {{
      if (window.MathJax && window.MathJax.typesetPromise) {{
        window.MathJax.typesetPromise().catch(() => null);
      }}
    }});

    function closeAsyOverlay() {{
      if (!asyOverlay || !asyOverlayContent) return;
      asyOverlay.dataset.open = "false";
      asyOverlay.hidden = true;
      asyOverlayContent.innerHTML = "";
    }}

    asyExpandButtons.forEach((button) => {{
      button.addEventListener("click", () => {{
        if (!asyOverlay || !asyOverlayContent) return;
        const figure = button.closest(".asy-diagram");
        const svg = figure ? figure.querySelector(".asy-diagram__svg") : null;
        if (!svg) return;
        const clone = svg.cloneNode(true);
        clone.removeAttribute("width");
        clone.removeAttribute("height");
        asyOverlayContent.innerHTML = "";
        asyOverlayContent.appendChild(clone);
        asyOverlay.hidden = false;
        asyOverlay.dataset.open = "true";
      }});
    }});

    if (asyOverlay) {{
      asyOverlay.addEventListener("click", (event) => {{
        if (event.target === asyOverlay || event.target.dataset.closeAsy === "true") {{
          closeAsyOverlay();
        }}
      }});
    }}

    if (asyOverlayClose) {{
      asyOverlayClose.addEventListener("click", closeAsyOverlay);
    }}

    document.addEventListener("keydown", (event) => {{
      if (event.key === "Escape" && asyOverlay && asyOverlay.dataset.open === "true") {{
        closeAsyOverlay();
      }}
    }});

    detailsBlocks.forEach((detail) => {{
      detail.addEventListener("toggle", () => {{
        if (detail.open && window.MathJax && window.MathJax.typesetPromise) {{
          window.MathJax.typesetPromise([detail]).catch(() => null);
        }}
      }});
    }});
  </script>
</body>
</html>
"""

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write(html)


def generate_report(input_path, output_path=None, title=None, run_benchmark=None, aggregate_benchmark=None):
    records, skipped_rows = load_records(input_path)
    if output_path is None:
        root, _ = os.path.splitext(input_path)
        output_path = f"{root}_report.html"
    if title is None:
        title = f"MathSensei"

    render_html_report(
        records,
        output_path,
        title=title,
        source_path=os.path.abspath(input_path),
        run_benchmark=run_benchmark,
        aggregate_benchmark=aggregate_benchmark,
        skipped_rows=skipped_rows,
    )
    return output_path, len(records)


def parse_args():
    parser = argparse.ArgumentParser(description="Render a presentation-friendly HTML report for MathSensei outputs.")
    parser.add_argument("--input", required=True, help="Path to a readable JSONL or raw cache JSONL file.")
    parser.add_argument("--output", help="Destination HTML file. Defaults to <input>_report.html.")
    parser.add_argument("--title", help="Custom report title.")
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    output_path, total = generate_report(cli_args.input, cli_args.output, cli_args.title)
    print(f"Rendered {total} records to {output_path}")

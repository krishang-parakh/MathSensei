import argparse
import json
import os
import re
import ast
from collections import Counter
from datetime import datetime
from html import escape


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


def extract_boxed_text(text):
    if not text:
        return None

    marker = "\\boxed{"
    start = text.rfind(marker)
    if start == -1:
        return None

    idx = start + len(marker)
    depth = 1
    collected = []

    while idx < len(text):
        char = text[idx]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return "".join(collected).strip()
        collected.append(char)
        idx += 1

    return None


def _format_option_answer(record, option_key):
    if not option_key:
        return None

    normalized_key = str(option_key).strip().upper()
    for option in record.get("options") or []:
        if str(option.get("key", "")).strip().upper() == normalized_key:
            label = clean_display_text(option.get("label") or "").strip()
            return f"{normalized_key}. {label}" if label else normalized_key
    return normalized_key


def infer_final_answer(record):
    explicit = record.get("final_answer") or record.get("answer")
    solution = record.get("final_generated_solution") or record.get("solution")
    inferred_option = infer_correct_option(explicit) or infer_correct_option(solution)

    if inferred_option and record.get("options"):
        return _format_option_answer(record, inferred_option)
    if inferred_option and not explicit:
        return inferred_option

    if explicit:
        explicit_text = clean_display_text(explicit).strip()

        gsm_match = re.search(r"####\s*([^\n]+)", explicit_text)
        if gsm_match:
            return gsm_match.group(1).strip()

        phrase_match = re.search(r"(?:the answer is|answer:)\s*([A-Za-z0-9\-\+\*/\^\(\)\[\]\{\}\\\., ]+)$", explicit_text, re.IGNORECASE)
        if phrase_match and len(explicit_text.splitlines()) == 1:
            candidate = phrase_match.group(1).strip().strip(".")
            if candidate:
                return candidate

        return explicit_text

    boxed = extract_boxed_text(solution)
    if boxed:
        return clean_display_text(boxed)

    if solution:
        solution_text = clean_display_text(solution)
        gsm_match = re.search(r"####\s*([^\n]+)", solution_text)
        if gsm_match:
            return gsm_match.group(1).strip()

    if solution:
        lines = [line.strip("- ").strip() for line in clean_display_text(solution).splitlines() if line.strip()]
        for line in reversed(lines):
            lowered = line.lower()
            if "final answer" in lowered or "therefore" in lowered:
                return line
        if lines:
            return lines[-1]

    program_output = record.get("program_output") or record.get("program_executor:output")
    if program_output:
        lines = [line.strip() for line in clean_display_text(program_output).splitlines() if line.strip()]
        if lines:
            return lines[-1]

    return None


def infer_correct_option(text):
    if not text:
        return None

    cleaned = clean_display_text(text)
    cleaned = cleaned.replace("**", "").replace("__", "").replace("`", "")
    patterns = [
        r"\bAnswer\s*[: ]\s*([A-E])\b",
        r"\bOption\s*([A-E])\b",
        r"\bthe answer is\s*([A-E])\b",
        r"\bTherefore,\s*the answer is\s*([A-E])\b",
        r"\bthe correct answer is\s*([A-E])\b",
        r"\bthe correct option(?: corresponding to [^.]*)?\s+is\s*([A-E])\b",
        r"\bmatches option\s*[\"']?([A-E])\b",
        r"\boption\s*[\"']?([A-E])[\)\.]",
    ]
    for pattern in patterns:
        match = re.search(pattern, cleaned, re.IGNORECASE)
        if match:
            return match.group(1).upper()
    return None


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

        normalized = {
            "pid": record.get("pid"),
            "problem": example.get("problem") or example.get("question") or example.get("Question"),
            "problem_type": example.get("type") or example.get("subject"),
            "level": example.get("level"),
            "ground_truth_solution": example.get("solution"),
            "generated_program": record.get("program"),
            "program_output": record.get("program_executor:output"),
            "program_error": record.get("program_executor:error"),
            "wolfram_query": record.get("wolfram_alpha_search:input"),
            "wolfram_output": record.get("wolfram_alpha_search:output"),
            "final_generated_solution": record.get("solution"),
            "final_answer": record.get("answer"),
            "modules_used": record.get("modules"),
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
            "problem": record.get("problem"),
            "problem_type": record.get("problem_type"),
            "level": record.get("level"),
            "ground_truth_solution": record.get("ground_truth_solution"),
            "generated_program": record.get("generated_program"),
            "program_output": record.get("program_output"),
            "program_error": record.get("program_error"),
            "wolfram_query": record.get("wolfram_query"),
            "wolfram_output": record.get("wolfram_output"),
            "final_generated_solution": record.get("final_generated_solution"),
            "final_answer": record.get("final_answer"),
            "modules_used": record.get("modules_used"),
            "options": normalize_options(record.get("options")),
            "correct_option": clean_display_text(record.get("correct_option")) or infer_correct_option(record.get("final_answer")) or infer_correct_option(record.get("final_generated_solution")),
        }

    modules = normalized.get("modules_used")
    if isinstance(modules, str):
        normalized["modules_used"] = [modules]
    elif modules is None:
        normalized["modules_used"] = []

    normalized["problem_type"] = normalized.get("problem_type") or "Unknown"
    normalized["level"] = normalized.get("level") or "Unknown"
    stem, embedded_options = parse_embedded_options(normalized.get("problem") or "(Problem text unavailable)")
    normalized["problem"] = clean_display_text(stem or "(Problem text unavailable)")
    if not normalized.get("options") and embedded_options:
        normalized["options"] = normalize_options(embedded_options)
    normalized["ground_truth_solution"] = clean_display_text(normalized.get("ground_truth_solution"))
    normalized["generated_program"] = clean_display_text(normalized.get("generated_program"))
    normalized["program_output"] = clean_display_text(normalized.get("program_output"))
    normalized["program_error"] = clean_display_text(normalized.get("program_error"))
    normalized["wolfram_query"] = clean_display_text(normalized.get("wolfram_query"))
    normalized["wolfram_output"] = clean_display_text(normalized.get("wolfram_output"))
    normalized["final_generated_solution"] = clean_display_text(normalized.get("final_generated_solution"))
    normalized["correct_option"] = clean_display_text(normalized.get("correct_option"))
    normalized["final_answer"] = infer_final_answer(normalized)
    normalized["status"] = classify_record(normalized)

    return normalized


def classify_record(record):
    program_error = (record.get("program_error") or "").strip()
    if program_error:
        return "needs-review"

    lowered = " ".join(
        str(record.get(field) or "").lower()
        for field in ("wolfram_output", "final_generated_solution", "program_output")
    )
    if "execution failed" in lowered or "cannot handle query" in lowered or "traceback" in lowered:
        return "needs-review"

    if record.get("final_answer"):
        return "complete"

    return "partial"


def load_records(input_path):
    records = []
    with open(input_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(normalize_record(json.loads(line)))
    return records


def build_summary(records):
    total = len(records)
    complete = sum(1 for record in records if record["status"] == "complete")
    partial = sum(1 for record in records if record["status"] == "partial")
    needs_review = sum(1 for record in records if record["status"] == "needs-review")
    with_program = sum(1 for record in records if record.get("generated_program"))
    with_wolfram = sum(1 for record in records if record.get("wolfram_query"))

    type_counts = Counter(record["problem_type"] for record in records)
    level_counts = Counter(record["level"] for record in records)

    return {
        "total": total,
        "complete": complete,
        "partial": partial,
        "needs_review": needs_review,
        "with_program": with_program,
        "with_wolfram": with_wolfram,
        "types": dict(sorted(type_counts.items(), key=lambda item: (-item[1], item[0]))),
        "levels": dict(sorted(level_counts.items(), key=lambda item: item[0])),
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


def _format_modules(modules):
    if not modules:
        return "Not captured"
    return ", ".join(str(module) for module in modules)


def _looks_like_math(text):
    if not text:
        return False

    candidate = str(text).strip()
    if not candidate:
        return False

    if any(token in candidate for token in ("$", "\\(", "\\)", "\\[", "\\]")):
        return False

    math_markers = ("\\frac", "\\sqrt", "\\boxed", "^", "_", "{", "}", "=", "\\cdot")
    if any(marker in candidate for marker in math_markers):
        return True

    # Short symbolic answers like [0,1), x+2, 48/95, A, or 3pi.
    return bool(re.fullmatch(r"[\[\]\(\)\{\}0-9A-Za-z,\.\-+/=*<>≤≥\s]+", candidate)) and len(candidate.split()) <= 4


def _strip_display_markdown(text):
    if text is None:
        return None

    cleaned = str(text)
    cleaned = cleaned.replace("**", "")
    cleaned = cleaned.replace("__", "")
    return cleaned


def _render_text_html(text, wrap_math=False, preserve_breaks=True):
    if text is None:
        return ""

    display_text = _strip_display_markdown(text)
    rendered = escape(str(display_text))
    if wrap_math and _looks_like_math(text):
        rendered = f"\\({rendered}\\)"
    if preserve_breaks:
        return rendered.replace("\n", "<br>")
    return rendered.replace("\n", " ")


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

    tag = "pre" if code else "div"
    class_name = "code-block" if code else "rich-text"
    return f"""
    <details class="detail-block">
      <summary>{escape(title)}</summary>
      <{tag} class="{class_name}">{escape(str(value)) if code else _render_text_html(value, preserve_breaks=True)}</{tag}>
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


def _report_context(records, dataset):
    problem_types = _distinct_known_values(records, lambda record: record.get("problem_type"))
    levels = _distinct_known_values(records, lambda record: record.get("level"))
    tooling = _distinct_known_values(records, _tooling_filter_value)
    modules = _distinct_known_values(records, lambda record: _format_modules(record.get("modules_used")))
    program_states = _distinct_known_values(records, _program_status_value)
    statuses = _distinct_known_values(records, lambda record: record.get("status"))

    return {
        "dataset": dataset,
        "show_problem_type": len(problem_types) > 1,
        "show_level": len(levels) > 1,
        "show_tools": len(tooling) > 1,
        "show_modules_meta": len(modules) > 1 and not (len(modules) == 1 and modules[0] == "Not captured"),
        "show_program_meta": len(program_states) > 1,
        "show_status_chip": len(statuses) > 1,
    }


def _dataset_badges(record, context):
    dataset = context["dataset"]
    badges = [f"Problem {_display_problem_number(record.get('pid'))}"]

    if dataset in {"MATH", "MMLU"} and context["show_problem_type"] and record.get("problem_type") != "Unknown":
        badges.append(str(record["problem_type"]))
    if dataset == "MATH" and context["show_level"] and record.get("level") != "Unknown":
        badges.append(str(record["level"]))
    if context["show_tools"]:
        if record.get("generated_program"):
            badges.append("Python")
        if record.get("wolfram_query"):
            badges.append("Wolfram")

    return badges


def _tooling_filter_value(record):
    tools = []
    if record.get("generated_program"):
        tools.append("Python")
    if record.get("wolfram_query"):
        tools.append("Wolfram")
    if not tools:
        return "No Tools"
    return " + ".join(tools)


def _program_status_value(record):
    if record.get("program_error"):
        return "Error"
    if record.get("generated_program"):
        return "OK"
    return "Not used"


def _distribution_values(records, value_fn):
    counts = Counter()
    for record in records:
        value = value_fn(record)
        if value in (None, "", "Unknown"):
            continue
        counts[str(value)] += 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def _dataset_distribution_sections(dataset, summary, records):
    if dataset == "MMLU":
        return [_maybe_distribution("Subjects", _distribution_values(records, lambda record: record.get("problem_type")))]

    if dataset == "MATH":
        sections = [_maybe_distribution("Problem Types", summary["types"])]
        sections.append(_maybe_distribution("Levels", summary["levels"]))
        return sections

    return []


def _dataset_filter_config(dataset, records):
    configs = []

    if dataset == "MATH":
        configs = [
            {
                "id": "primaryFilter",
                "label": "Problem Type",
                "attr": "primary",
                "options": sorted({record["problem_type"] for record in records if record["problem_type"] != "Unknown"}),
            },
            {
                "id": "secondaryFilter",
                "label": "Level",
                "attr": "secondary",
                "options": sorted({record["level"] for record in records if record["level"] != "Unknown"}),
            },
        ]
    elif dataset == "MMLU":
        configs = [
            {
                "id": "primaryFilter",
                "label": "Subject",
                "attr": "primary",
                "options": sorted({record["problem_type"] for record in records if record["problem_type"] != "Unknown"}),
            },
        ]
    else:
        configs = [
            {
                "id": "primaryFilter",
                "label": "Problem Type",
                "attr": "primary",
                "options": sorted({record["problem_type"] for record in records if record["problem_type"] != "Unknown"}),
            },
            {
                "id": "secondaryFilter",
                "label": "Level",
                "attr": "secondary",
                "options": sorted({record["level"] for record in records if record["level"] != "Unknown"}),
            },
        ]

    return [config for config in configs if len(config["options"]) > 1]


def _record_filter_values(record, dataset):
    if dataset == "MATH":
        return record["problem_type"], record["level"]
    if dataset == "MMLU":
        return record["problem_type"], record["level"]
    return record["problem_type"], record["level"]


def _render_option_board(record):
    options = record.get("options") or []
    if not options:
        return ""

    option_cards = []
    correct_option = (record.get("correct_option") or "").upper()
    for option in options:
        key = option.get("key", "")
        label = option.get("label", "")
        classes = ["quiz-option"]
        if correct_option and key.upper() == correct_option:
            classes.append("quiz-option--correct")
        option_cards.append(
            f"""
            <div class="{' '.join(classes)}">
              <span class="quiz-option__key">{escape(key)}</span>
              <span class="quiz-option__text">{_render_text_html(label, preserve_breaks=False)}</span>
            </div>
            """
        )

    legend = ""
    if correct_option:
        legend = f'<div class="quiz-legend">Correct choice: <strong>{escape(correct_option)}</strong></div>'

    return f"""
    <section class="quiz-board">
      <div class="quiz-board__header">
        <span class="quiz-board__label">Options</span>
        {legend}
      </div>
      <div class="quiz-grid">
        {''.join(option_cards)}
      </div>
    </section>
    """


def _record_card(record, context):
    dataset = context["dataset"]
    status_labels = {
        "complete": "Complete",
        "partial": "Partial",
        "needs-review": "Needs Review",
    }
    answer = record.get("final_answer") or "Answer not extracted"
    tags = _dataset_badges(record, context)
    primary_filter, secondary_filter = _record_filter_values(record, dataset)
    option_board = _render_option_board(record)

    detail_blocks = [
        _detail_block("Model Solution", record.get("final_generated_solution")),
        _detail_block("Ground Truth Solution", record.get("ground_truth_solution")),
        _detail_block("Generated Program", record.get("generated_program"), code=True),
        _detail_block("Program Output", record.get("program_output"), code=True),
        _detail_block("Program Error", record.get("program_error"), code=True),
        _detail_block("Wolfram Query", record.get("wolfram_query"), code=True),
        _detail_block("Wolfram Output", record.get("wolfram_output")),
    ]

    meta_items = []
    if context["show_modules_meta"]:
        meta_items.append(
            f"""
            <div>
              <span class="meta-label">{'Modules' if dataset == 'MATH' else 'Tools Used'}</span>
              <div class="meta-value">{escape(_format_modules(record.get('modules_used')))}</div>
            </div>
            """
        )
    if context["show_program_meta"]:
        meta_items.append(
            f"""
            <div>
              <span class="meta-label">{'Program Status' if dataset in ('MATH', 'GSM') else 'Execution Status'}</span>
              <div class="meta-value">{escape(_program_status_value(record))}</div>
            </div>
            """
        )

    status_chip_html = ""
    if context["show_status_chip"]:
        status_chip_html = f'<span class="status-chip status-chip--{record["status"]}">{status_labels[record["status"]]}</span>'

    return f"""
    <article class="problem-card" data-status="{record['status']}" data-type="{escape(record['problem_type'])}" data-level="{escape(record['level'])}" data-primary="{escape(str(primary_filter))}" data-secondary="{escape(str(secondary_filter))}">
      <div class="problem-card__header">
        <div class="problem-card__badges">
          {''.join(f'<span class="tag">{escape(tag)}</span>' for tag in tags)}
        </div>
        {status_chip_html}
      </div>
      <section class="question-shell">
        <div class="problem-card__title">{_render_text_html(record['problem'], preserve_breaks=True)}</div>
      </section>
      {option_board}
      <div class="answer-card">
        <span class="answer-label">Final Answer</span>
        <div class="answer-value">{_render_text_html(answer, wrap_math=True, preserve_breaks=False)}</div>
      </div>
      {'<div class="meta-grid">' + ''.join(meta_items) + '</div>' if meta_items else ''}
      <div class="detail-stack">
        {''.join(detail_blocks)}
      </div>
    </article>
    """


def render_html_report(records, output_path, title, source_path=None):
    summary = build_summary(records)
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    dataset = infer_dataset(records=records, source_path=source_path, title=title)
    context = _report_context(records, dataset)
    cards_html = "".join(_record_card(record, context) for record in records)
    distribution_sections = [section for section in _dataset_distribution_sections(dataset, summary, records) if section]
    filter_config = _dataset_filter_config(dataset, records)
    filter_controls_html = "".join(
        f"""
        <div class="control">
          <label for="{escape(config['id'])}">{escape(config['label'])}</label>
          <select id="{escape(config['id'])}" data-attr="{escape(config['attr'])}">
            <option value="">All {escape(config['label']).lower()}</option>
            {''.join(f'<option value="{escape(str(option))}">{escape(str(option))}</option>' for option in config['options'])}
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
            <option value="partial">Partial</option>
            <option value="needs-review">Needs Review</option>
          </select>
        </div>
        """

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
      width: min(1380px, calc(100vw - 32px));
      margin: 28px auto 52px;
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
      padding: 34px;
      backdrop-filter: blur(18px) saturate(130%);
      position: relative;
      overflow: hidden;
    }}

    .hero::before {{
      content: "";
      position: absolute;
      inset: 0;
      background: linear-gradient(120deg, rgba(96,165,250,0.10), transparent 24%, transparent 76%, rgba(94,234,212,0.08));
      pointer-events: none;
    }}

    .hero h1 {{
      margin: 0;
      font-family: var(--display);
      font-size: clamp(2.1rem, 4.8vw, 4rem);
      line-height: 1.04;
      letter-spacing: -0.04em;
      max-width: 11ch;
      position: relative;
      z-index: 1;
      text-wrap: balance;
    }}

    .hero p {{
      margin: 14px 0 0;
      color: var(--muted);
      max-width: 72ch;
      line-height: 1.72;
      font-size: 1rem;
      position: relative;
      z-index: 1;
    }}

    .hero-meta {{
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
      margin-top: 22px;
      position: relative;
      z-index: 1;
    }}

    .hero-meta span {{
      padding: 10px 14px;
      background: rgba(255, 255, 255, 0.045);
      border: 1px solid rgba(255, 255, 255, 0.1);
      border-radius: 999px;
      color: var(--muted);
      font-size: 0.94rem;
      box-shadow: inset 0 1px 0 rgba(255,255,255,0.04);
    }}

    .summary-grid,
    .panel-grid,
    .problem-grid {{
      display: grid;
      gap: 18px;
    }}

    .summary-grid {{
      grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
      margin-top: 22px;
    }}

    .panel-grid {{
      grid-template-columns: 1.2fr 1fr 1fr;
      margin-top: 18px;
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
      padding: 22px;
      position: relative;
      overflow: hidden;
      isolation: isolate;
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
      font-size: 0.78rem;
      display: block;
      margin-bottom: 12px;
      letter-spacing: 0.14em;
      text-transform: uppercase;
    }}

    .summary-card .value {{
      font-family: var(--display);
      font-size: clamp(2rem, 3.5vw, 2.9rem);
      font-weight: 700;
      letter-spacing: -0.04em;
      line-height: 1;
      background: linear-gradient(135deg, #f8fbff 0%, #7dd3fc 50%, #5eead4 100%);
      -webkit-background-clip: text;
      background-clip: text;
      color: transparent;
    }}

    .summary-card .caption {{
      color: var(--muted);
      margin-top: 10px;
      font-size: 0.92rem;
      line-height: 1.5;
    }}

    .panel {{
      padding: 22px;
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
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 14px;
      margin: 24px 0 20px;
      padding: 16px;
      background: rgba(9, 17, 28, 0.78);
      border: 1px solid rgba(255, 255, 255, 0.07);
      border-radius: 24px;
      box-shadow: var(--shadow-soft);
      position: sticky;
      top: 16px;
      z-index: 5;
      backdrop-filter: blur(22px) saturate(135%);
    }}

    .control {{
      display: flex;
      flex-direction: column;
      gap: 8px;
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
      border-radius: 16px;
      border: 1px solid rgba(255,255,255,0.09);
      background:
        linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.015)),
        var(--panel-strong);
      padding: 14px 15px;
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
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      align-items: start;
    }}

    .problem-card {{
      padding: 22px;
      position: relative;
      overflow: hidden;
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
      transform: translateY(-4px);
      border-color: rgba(94, 234, 212, 0.18);
      box-shadow: 0 22px 70px rgba(0,0,0,0.42);
    }}

    .problem-card__header {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: flex-start;
    }}

    .problem-card__badges {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      padding-right: 8px;
    }}

    .tag,
    .status-chip {{
      border-radius: 999px;
      padding: 8px 12px;
      font-size: 0.78rem;
      border: 1px solid rgba(255,255,255,0.09);
      backdrop-filter: blur(14px);
    }}

    .tag {{
      background: rgba(255, 255, 255, 0.045);
      color: #bfd0e7;
      box-shadow: inset 0 1px 0 rgba(255,255,255,0.03);
    }}

    .status-chip--complete {{
      background: linear-gradient(135deg, rgba(94, 234, 212, 0.18), rgba(34, 197, 94, 0.08));
      color: var(--accent);
      border-color: rgba(94, 234, 212, 0.32);
    }}

    .status-chip--partial {{
      background: linear-gradient(135deg, rgba(251,191,36,0.14), rgba(251,146,60,0.08));
      color: var(--warning);
      border-color: rgba(251, 191, 36, 0.26);
    }}

    .status-chip--needs-review {{
      background: linear-gradient(135deg, rgba(251,113,133,0.16), rgba(244,114,182,0.08));
      color: var(--danger);
      border-color: rgba(251, 113, 133, 0.28);
    }}

    .question-shell {{
      margin: 18px 0 16px;
      padding: 18px 20px;
      border-radius: 22px;
      border: 1px solid rgba(110, 168, 255, 0.14);
      background:
        radial-gradient(circle at top right, rgba(94, 234, 212, 0.08), transparent 34%),
        linear-gradient(160deg, rgba(255,255,255,0.03), rgba(255,255,255,0.015));
      box-shadow: inset 0 1px 0 rgba(255,255,255,0.04);
    }}

    .problem-card__title {{
      margin: 0;
      font-family: var(--display);
      font-size: 1.18rem;
      line-height: 1.52;
      font-weight: 700;
      overflow-wrap: anywhere;
      letter-spacing: -0.015em;
      color: #f8fbff;
    }}

    .quiz-board {{
      margin: 0 0 18px;
      padding: 8px 0 0;
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
      font-size: 0.78rem;
      color: #a9c6ef;
      letter-spacing: 0.2em;
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
      min-height: 58px;
      padding: 14px 16px;
      display: flex;
      align-items: flex-start;
      gap: 12px;
      color: #edf4ff;
      background:
        linear-gradient(180deg, rgba(19, 31, 54, 0.96), rgba(15, 25, 46, 0.98));
      border: 1px solid rgba(127, 167, 245, 0.16);
      border-radius: 18px;
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
      line-height: 1.5;
      font-size: 0.98rem;
      padding-top: 2px;
    }}

    .answer-card {{
      padding: 18px;
      border-radius: var(--radius-sm);
      background:
        radial-gradient(circle at top right, rgba(94,234,212,0.11), transparent 34%),
        linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.015));
      border: 1px solid rgba(94, 234, 212, 0.16);
      box-shadow: inset 0 1px 0 rgba(255,255,255,0.04);
    }}

    .answer-label {{
      display: block;
      color: var(--muted);
      font-size: 0.78rem;
      text-transform: uppercase;
      letter-spacing: 0.16em;
      margin-bottom: 10px;
    }}

    .answer-value {{
      font-size: 1.15rem;
      line-height: 1.55;
      word-break: break-word;
      overflow-wrap: anywhere;
      color: #f8fbff;
    }}

    .meta-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 14px;
      margin-top: 18px;
    }}

    .meta-label {{
      display: block;
      color: var(--muted);
      font-size: 0.74rem;
      text-transform: uppercase;
      letter-spacing: 0.16em;
      margin-bottom: 8px;
    }}

    .meta-value {{
      font-size: 0.96rem;
      line-height: 1.45;
      color: #dbe7f6;
    }}

    .detail-stack {{
      display: grid;
      gap: 12px;
      margin-top: 18px;
    }}

    .detail-block {{
      border: 1px solid rgba(255,255,255,0.08);
      border-radius: 18px;
      background: rgba(255, 255, 255, 0.03);
      overflow: hidden;
      transition: border-color 180ms ease, background 180ms ease;
    }}

    .detail-block[open] {{
      border-color: rgba(96, 165, 250, 0.22);
      background: rgba(255,255,255,0.04);
    }}

    .detail-block summary {{
      list-style: none;
      cursor: pointer;
      padding: 14px 16px;
      font-weight: 600;
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

    .rich-text,
    .code-block {{
      margin: 0;
      padding: 16px;
      line-height: 1.6;
      white-space: pre-wrap;
      word-break: break-word;
      overflow-wrap: anywhere;
    }}

    .code-block {{
      font-family: var(--mono);
      font-size: 0.89rem;
      background: rgba(4, 10, 18, 0.92);
      color: #d8e6ff;
      border-top: 1px solid rgba(255,255,255,0.04);
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

    @media (max-width: 1080px) {{
      .panel-grid {{
        grid-template-columns: 1fr;
      }}
    }}

    @media (max-width: 820px) {{
      .controls {{
        grid-template-columns: 1fr;
      }}

      .quiz-grid {{
        grid-template-columns: 1fr;
      }}

      .meta-grid {{
        grid-template-columns: 1fr;
      }}

      .shell {{
        width: min(100vw - 20px, 1380px);
        margin: 10px auto 28px;
      }}

      .hero,
      .problem-card,
      .panel,
      .summary-card {{
        border-radius: 20px;
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
      <h1>{escape(title)}</h1>
      <p>Use the filters below to focus on a topic, level, or review status, then open any problem card to inspect the generated reasoning, code, and tool outputs.</p>
      <div class="summary-grid">
        <div class="summary-card">
          <span class="label">Problems</span>
          <div class="value">{summary['total']}</div>
          <div class="caption">Total records in this run</div>
        </div>
        <div class="summary-card">
          <span class="label">Complete</span>
          <div class="value">{summary['complete']}</div>
          <div class="caption">Answer extracted cleanly</div>
        </div>
        <div class="summary-card">
          <span class="label">Partial</span>
          <div class="value">{summary['partial']}</div>
          <div class="caption">Useful output, but answer needs confirmation</div>
        </div>
        <div class="summary-card">
          <span class="label">Needs Review</span>
          <div class="value">{summary['needs_review']}</div>
          <div class="caption">Execution or parsing issue detected</div>
        </div>
        <div class="summary-card">
          <span class="label">Python Used</span>
          <div class="value">{summary['with_program']}</div>
          <div class="caption">Records with generated code</div>
        </div>
        <div class="summary-card">
          <span class="label">Wolfram Used</span>
          <div class="value">{summary['with_wolfram']}</div>
          <div class="caption">Records with Wolfram queries</div>
        </div>
      </div>
    </section>

    <section class="panel-grid">
      {distribution_html}
      <section class="panel">
        <h3>How To Use</h3>
        <p class="muted">Search by keyword, use the filters only when they add real signal, and open any card to show either the polished solution or the supporting tool trace depending on your audience.</p>
      </section>
    </section>

    <section class="controls">
      <div class="control">
        <label for="searchBox">Search</label>
        <input id="searchBox" type="search" placeholder="Search problems, answers, or solutions">
      </div>
      {filter_controls_html}
      {status_filter_html}
    </section>

    <section id="problemGrid" class="problem-grid">
      {cards_html}
    </section>
    <div id="emptyState" class="empty-state">No problems match the current filters.</div>
  </main>

  <script>
    window.MathJax = {{
      tex: {{
        inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
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
    const detailsBlocks = Array.from(document.querySelectorAll(".detail-block"));

    function applyFilters() {{
      const query = searchBox.value.trim().toLowerCase();
      const selectedStatus = statusFilter ? statusFilter.value : "";

      let visibleCount = 0;

      cards.forEach((card) => {{
        const matchesQuery = !query || card.textContent.toLowerCase().includes(query);
        const matchesStatus = !selectedStatus || card.dataset.status === selectedStatus;
        const matchesDatasetFilters = datasetFilters.every((filter) => {{
          const attr = filter.dataset.attr;
          const value = filter.value;
          if (!value) return true;
          return card.dataset[attr] === value;
        }});
        const visible = matchesQuery && matchesStatus && matchesDatasetFilters;
        card.style.display = visible ? "" : "none";
        if (visible) visibleCount += 1;
      }});

      emptyState.style.display = visibleCount === 0 ? "block" : "none";
    }}

    [searchBox, statusFilter, ...datasetFilters].filter(Boolean).forEach((element) => {{
      element.addEventListener("input", applyFilters);
      element.addEventListener("change", applyFilters);
    }});

    window.addEventListener("load", () => {{
      if (window.MathJax && window.MathJax.typesetPromise) {{
        window.MathJax.typesetPromise().catch(() => null);
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

    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write(html)


def generate_report(input_path, output_path=None, title=None):
    records = load_records(input_path)
    if output_path is None:
        root, _ = os.path.splitext(input_path)
        output_path = f"{root}_report.html"
    if title is None:
        title = f"MathSensei"

    render_html_report(records, output_path, title=title, source_path=os.path.abspath(input_path))
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

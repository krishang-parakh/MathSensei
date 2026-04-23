import os
import textwrap


GLOBAL_MODEL_OVERRIDES = {
    "gemini": {
        "python_model": "gemini",
        "knowledge_model": "gemini",
        "bing_model": "gemini",
        "sg_model": "gemini",
        "wolfram_model": "gemini",
    }
}

GLOBAL_MODEL_CHOICES = ["no"] + sorted(GLOBAL_MODEL_OVERRIDES.keys())


def apply_global_model_overrides(args):
    global_model = getattr(args, "global_model", "no")
    if not global_model or global_model == "no":
        return args

    overrides = GLOBAL_MODEL_OVERRIDES.get(global_model, {})
    for attr_name, override_value in overrides.items():
        if getattr(args, attr_name, "no") == "no":
            setattr(args, attr_name, override_value)
    return args


def solution_prompt_family(model_name):
    if model_name == "cot":
        return "cot"
    if model_name == "pot":
        return "pot"
    if model_name == "kr_sg":
        return "kr_sg"
    if model_name == "kr_pg_sg":
        return "kr_pg_sg"
    if model_name == "kr_pg_walpha_sg":
        return "kr_walpha_sg"
    return "kr_walpha_sg"


LINE_WIDTH = 88


def _line(char="="):
    return char * LINE_WIDTH


def _compact(value, width=72):
    if value is None:
        return "-"

    text = str(value).strip()
    if not text:
        return "-"

    text = " ".join(text.split())
    if len(text) <= width:
        return text
    return text[: width - 3] + "..."


def _print_kv(label, value):
    try:
        print(f"{label:<14}: {value}")
    except UnicodeEncodeError:
        # Windows consoles can still default to legacy codepages (e.g. cp1252).
        # Fall back to an ASCII-safe representation so runs don't crash on symbols like \u2220.
        safe = f"{label:<14}: {value}"
        safe = safe.encode("ascii", "backslashreplace").decode("ascii")
        print(safe)


def _safe_print(value=""):
    try:
        print(value)
    except UnicodeEncodeError:
        safe = str(value).encode("ascii", "backslashreplace").decode("ascii")
        print(safe)


def _dataset_label(value):
    if value == "ALL":
        return "Mixed"
    return value


def _format_accuracy(value):
    if value is None:
        return "-"
    return f"{value * 100:.2f}%"


def module_status(output):
    if output is None:
        return "empty"

    text = str(output).strip()
    if not text:
        return "empty"

    lowered = text.lower()
    if "execution failed" in lowered or "cannot handle query" in lowered or "error" in lowered:
        return "error"

    return "ok"


def print_run_header(args, result_file, test_number, data_note=None):
    _safe_print()
    _safe_print(_line("="))
    _safe_print("MathSensei Run")
    _safe_print(_line("="))
    _print_kv("Dataset", _dataset_label(args.dataset))
    _print_kv("Model", args.model)
    _print_kv("Label", args.label)
    _print_kv("Split", args.test_split)
    _print_kv("Examples", test_number)
    _print_kv("Output", result_file)
    if data_note:
        _print_kv("Data Note", data_note)
    _safe_print(_line("-"))


def print_problem_header(pid, total, preview):
    _safe_print()
    _safe_print(_line("-"))
    _safe_print(f"Problem {pid + 1}/{total}")
    _safe_print(_line("-"))
    wrapped = textwrap.fill(_compact(preview, width=220), width=LINE_WIDTH)
    _safe_print(wrapped)


def print_module_plan(modules):
    _safe_print(f"Modules       : {' -> '.join(modules)}")


def print_module_result(module_name, module_input, module_output):
    _safe_print()
    _safe_print(f"[{module_name}]")
    _print_kv("Status", module_status(module_output))
    _print_kv("Input", _compact(module_input))
    _print_kv("Output", _compact(module_output))


def print_problem_summary(cache):
    _safe_print()
    _safe_print("Problem Summary")
    _safe_print(_line("-"))
    _print_kv("Dataset", _dataset_label(cache.get("dataset")))
    _print_kv("Final Answer", _compact(cache.get("answer")))
    _print_kv("WA Query", _compact(cache.get("wolfram_alpha_search:input")))
    _print_kv("WA Output", _compact(cache.get("wolfram_alpha_search:output")))
    _print_kv("Py Error", _compact(cache.get("program_executor:error")))
    _print_kv("Modules", _compact(cache.get("modules"), width=220))


def print_run_summary(total, succeeded, failed, result_root, report_path=None, benchmark=None):
    _safe_print()
    _safe_print(_line("="))
    _safe_print("Run Summary")
    _safe_print(_line("="))
    _print_kv("Total", total)
    _print_kv("Succeeded", succeeded)
    _print_kv("Failed", failed)
    if benchmark:
        _print_kv("Evaluated", benchmark.get("evaluated"))
        _print_kv("Correct", benchmark.get("correct"))
        _print_kv("Accuracy", _format_accuracy(benchmark.get("accuracy")))
        _print_kv("Model", benchmark.get("model"))
        _print_kv("Dataset", _dataset_label(benchmark.get("dataset")))
    _print_kv("Outputs", os.path.normpath(result_root))
    if report_path:
        _print_kv("Report", os.path.normpath(report_path))
    _safe_print(_line("="))


def print_warning(message):
    _safe_print(f"[WARN] {message}")

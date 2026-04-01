import os
import textwrap


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
    print(f"{label:<14}: {value}")


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


def print_run_header(args, result_file, test_number):
    print()
    print(_line("="))
    print("MathSensei Run")
    print(_line("="))
    _print_kv("Dataset", args.dataset)
    _print_kv("Model", args.model)
    _print_kv("Label", args.label)
    _print_kv("Split", args.test_split)
    _print_kv("Examples", test_number)
    _print_kv("Output", result_file)
    print(_line("-"))


def print_problem_header(pid, total, preview):
    print()
    print(_line("-"))
    print(f"Problem {pid + 1}/{total}")
    print(_line("-"))
    wrapped = textwrap.fill(_compact(preview, width=220), width=LINE_WIDTH)
    print(wrapped)


def print_module_plan(modules):
    print(f"Modules       : {' -> '.join(modules)}")


def print_module_result(module_name, module_input, module_output):
    print()
    print(f"[{module_name}]")
    _print_kv("Status", module_status(module_output))
    _print_kv("Input", _compact(module_input))
    _print_kv("Output", _compact(module_output))


def print_problem_summary(cache):
    print()
    print("Problem Summary")
    print(_line("-"))
    _print_kv("Final Answer", _compact(cache.get("answer")))
    _print_kv("WA Query", _compact(cache.get("wolfram_alpha_search:input")))
    _print_kv("WA Output", _compact(cache.get("wolfram_alpha_search:output")))
    _print_kv("Py Error", _compact(cache.get("program_executor:error")))
    _print_kv("Modules", _compact(cache.get("modules"), width=220))


def print_run_summary(total, succeeded, failed, result_root):
    print()
    print(_line("="))
    print("Run Summary")
    print(_line("="))
    _print_kv("Total", total)
    _print_kv("Succeeded", succeeded)
    _print_kv("Failed", failed)
    _print_kv("Outputs", os.path.normpath(result_root))
    print(_line("="))


def print_warning(message):
    print(f"[WARN] {message}")

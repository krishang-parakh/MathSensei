import re
import subprocess
import sys
import unicodedata


LEADING_NOISE_PREFIXES = (
    "question:",
    "modules used till now:",
    "python generator:",
    "python code:",
    "knowledge retrieval:",
    "bing search response:",
    "wolfram thought:",
    "wolfram_alpha response:",
    "wolfram alpha response:",
    "final query:",
    "solution:",
    "output:",
    "observation:",
)

INLINE_NOISE_PREFIXES = (
    "python code,",
    "make sure that",
    "do not include",
    "write only executable python code",
    "rules:",
    "errors fixed:",
)

TRAILING_SECTION_PREFIXES = (
    "python output:",
    "solution:",
    "output:",
    "error message:",
    "errors fixed:",
    "observation:",
    "question:",
)

COMMON_REPAIR_ERROR_SNIPPETS = (
    "list indices must be integers or slices, not symbol",
    "keyerror: 0",
    "keyerror:",
    "has no attribute 'evalf'",
    "has no attribute 'free_symbols'",
    "has no attribute 'as_real_imag'",
    "object is not iterable",
    "invalid syntax",
    "syntaxerror",
    "nameerror",
    "is not defined",
    "unexpected eof",
    "unterminated string",
    "eol while scanning string literal",
    "expected an indented block",
    "unmatched",
    "cannot determine truth value of relational",
    "can only solve for one symbol at a time",
    "sympifyerror",
    "list index out of range",
    "argument should be a string or a rational instance",
    "solve_univariate_inequality",
    "is_commutative",
    "none type",
    "produced no output",
)

PROSE_STEP_PREFIXES = (
    "define ",
    "equation ",
    "solve ",
    "output ",
    "substitute ",
    "calculate ",
    "compute ",
    "let ",
    "now ",
    "step ",
)

NAME_ERROR_IMPORT_FIXES = {
    "math": "import math",
    "radians": "from math import radians",
    "degrees": "from math import degrees",
    "isclose": "from math import isclose",
    "ceil": "from math import ceil",
    "floor": "from math import floor",
    "factorial": "from math import factorial",
    "comb": "from math import comb",
    "perm": "from math import perm",
    "gcd": "from math import gcd",
    "lcm": "from math import lcm",
    "prod": "from math import prod",
    "tau": "from math import tau",
    "e": "from math import e",
    "inf": "from math import inf",
    "Fraction": "from fractions import Fraction",
    "Decimal": "from decimal import Decimal",
}

SYMPY_NAME_FIXES = {
    "symbols",
    "Symbol",
    "Eq",
    "solve",
    "Rational",
    "Integer",
    "sqrt",
    "simplify",
    "factor",
    "expand",
    "Matrix",
    "pi",
    "E",
    "oo",
    "Abs",
    "S",
    "N",
    "nsimplify",
    "sin",
    "cos",
    "tan",
    "asin",
    "acos",
    "atan",
    "log",
    "diff",
    "integrate",
    "limit",
    "summation",
    "product",
    "solveset",
    "Interval",
    "Union",
    "FiniteSet",
    "latex",
}

AUTO_INSTALL_PACKAGE_MAP = {
    "sympy": "sympy",
    "numpy": "numpy",
    "scipy": "scipy",
    "matplotlib": "matplotlib",
    "pandas": "pandas",
    "mpmath": "mpmath",
    "networkx": "networkx",
    "sklearn": "scikit-learn",
    "skimage": "scikit-image",
    "PIL": "pillow",
    "cv2": "opencv-python",
    "bs4": "beautifulsoup4",
    "yaml": "PyYAML",
    "seaborn": "seaborn",
    "func_timeout": "func-timeout",
    "dotenv": "python-dotenv",
    "openai": "openai",
    "langchain": "langchain",
    "wolframalpha": "wolframalpha",
    "google.generativeai": "google-generativeai",
}


def _normalize_code_fences(text):
    normalized = str(text).replace("\r\n", "\n")
    normalized = normalized.strip().strip('"').strip("'")
    if normalized.startswith("```python"):
        normalized = normalized[len("```python"):].strip()
    elif normalized.startswith("```"):
        normalized = normalized[len("```"):].strip()
    if normalized.endswith("```"):
        normalized = normalized[:-3].strip()
    return normalized


def _normalize_unicode_in_code(text):
    if text is None:
        return ""

    normalized = unicodedata.normalize("NFKC", str(text))
    replacements = {
        "\u00A0": " ",
        "\u2013": "-",
        "\u2014": "-",
        "\u2212": "-",
        "\u2018": "'",
        "\u2019": "'",
        "\u201C": '"',
        "\u201D": '"',
        "\u2026": "...",
        "\u00BC": "1/4",
        "\u00BD": "1/2",
        "\u00BE": "3/4",
        "\u2153": "1/3",
        "\u2154": "2/3",
        "\u2155": "1/5",
        "\u2156": "2/5",
        "\u2157": "3/5",
        "\u2158": "4/5",
        "\u2159": "1/6",
        "\u215A": "5/6",
        "\u215B": "1/8",
        "\u215C": "3/8",
        "\u215D": "5/8",
        "\u215E": "7/8",
        "\u2044": "/",
        "\u2217": "*",
        "\u00B7": "*",
    }
    for old, new in replacements.items():
        normalized = normalized.replace(old, new)

    normalized = normalized.replace("\u200B", "").replace("\u200C", "").replace("\u200D", "")
    return normalized


def _is_noise_line(stripped_line):
    lowered = stripped_line.lower()
    if not lowered:
        return False
    if lowered.startswith("# python code"):
        return True
    if lowered.startswith(LEADING_NOISE_PREFIXES):
        return True
    return lowered.startswith(INLINE_NOISE_PREFIXES)


def _looks_like_plain_english_step_line(stripped_line):
    if not stripped_line or stripped_line.startswith("#"):
        return False

    if any(token in stripped_line for token in ("=", "(", ")", "[", "]", "{", "}", ".", "->")):
        return False

    if re.match(r"^(def|class|import|from|print|return|for|while|if|elif|else|try|except|with|lambda)\b", stripped_line):
        return False

    lowered = stripped_line.lower()
    if lowered.startswith(PROSE_STEP_PREFIXES):
        return True

    return bool(re.fullmatch(r"[A-Za-z][A-Za-z0-9 ,:'\"/\-]+", stripped_line))


def sanitize_generated_python(program):
    if not program:
        return ""

    text = _normalize_code_fences(program)
    text = _normalize_unicode_in_code(text)
    lines = text.splitlines()
    cleaned_lines = []
    saw_code = False

    for raw_line in lines:
        line = raw_line.rstrip()
        stripped = line.strip()
        lowered = stripped.lower()

        # Drop common "enumerated heading" prose from repair models (e.g. "1) Fix ...", "2. Fix ...").
        # These lines are not valid Python and can sneak past other heuristics due to parentheses/punctuation.
        if stripped and not stripped.startswith("#"):
            if re.match(r"^\s*\d+\)\s+\S", raw_line):
                continue
            if re.match(r"^\s*\d+\.\s+\S", raw_line):
                continue

        if stripped.startswith("Corrected Python Code:"):
            stripped = stripped.split(":", 1)[1].strip()
            line = stripped
            lowered = stripped.lower()
            if not stripped:
                continue

        if saw_code and lowered.startswith(TRAILING_SECTION_PREFIXES):
            break

        if stripped in {"```", "'''", '"""'}:
            continue

        if _is_noise_line(stripped) or _looks_like_plain_english_step_line(stripped):
            continue

        if not stripped and not saw_code:
            continue

        cleaned_lines.append(line)
        if stripped:
            saw_code = True

    return "\n".join(cleaned_lines).strip()


def _extract_missing_name(error_message):
    if not error_message:
        return None

    match = re.search(r"name\s+'([^']+)'\s+is not defined", str(error_message), re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None


def _prepend_import_if_missing(program, import_line):
    stripped_lines = [line.strip() for line in str(program).splitlines()]
    if import_line in stripped_lines:
        return str(program)
    return f"{import_line}\n{program}".strip()


def _insert_import_after_sympy_wildcard(program, import_line):
    lines = str(program).splitlines()
    if import_line in [line.strip() for line in lines]:
        return str(program)

    insert_at = None
    for index, line in enumerate(lines):
        if line.strip() == "from sympy import *":
            insert_at = index + 1

    if insert_at is None:
        return _prepend_import_if_missing(program, import_line)

    updated_lines = list(lines)
    updated_lines.insert(insert_at, import_line)
    return "\n".join(updated_lines).strip()


def _insert_dict_true_into_solve_call(line):
    if "solve(" not in line or "dict=True" in line:
        return None

    closing_index = line.rfind(")")
    opening_index = line.find("solve(")
    if opening_index == -1 or closing_index <= opening_index:
        return None

    return f"{line[:closing_index]}, dict=True{line[closing_index:]}"


def _repair_sympy_solve_output_shape(program, error_message):
    lowered_error = str(error_message or "").lower()
    shape_error = (
        "list indices must be integers or slices, not symbol" in lowered_error
        or "keyerror:" in lowered_error
        or lowered_error.strip() == "0"
    )
    if not shape_error:
        return None, None

    lines = str(program).splitlines()
    updated_lines = list(lines)
    changed = False

    solve_targets = set()
    for line in lines:
        stripped = line.strip()
        match = re.match(r"^([A-Za-z_]\w*)\s*=\s*solve\(", stripped)
        if match:
            solve_targets.add(match.group(1))

    updated_program = "\n".join(updated_lines).strip()
    for index, line in enumerate(lines):
        stripped = line.strip()
        match = re.match(r"^([A-Za-z_]\w*)\s*=\s*solve\(", stripped)
        if not match:
            continue
        target = match.group(1)
        if target and (
            re.search(rf"\b{re.escape(target)}\s*\[\s*0\s*\]\s*\[\s*[A-Za-z_]\w*\s*\]", updated_program)
            or re.search(rf"\b{re.escape(target)}\s*\[\s*[A-Za-z_]\w*\s*\]", updated_program)
        ):
            repaired_line = _insert_dict_true_into_solve_call(line)
            if repaired_line and repaired_line != line:
                updated_lines[index] = repaired_line
                changed = True

    updated_program = "\n".join(updated_lines).strip()
    for target in solve_targets:
        updated_program, replacements = re.subn(
            rf"\b{re.escape(target)}\s*\[\s*([A-Za-z_]\w*)\s*\]",
            lambda match: f"{target}[0][{match.group(1)}]",
            updated_program,
        )
        if replacements:
            changed = True

    if not changed:
        return None, None

    return (
        updated_program,
        "Adjusted SymPy solve() so the returned solution format uses dict=True and first-solution dict indexing for symbol-key lookups.",
    )


def _repair_integer_lcm_or_gcd(program, error_message):
    lowered_error = str(error_message or "").lower()
    if "is_commutative" not in lowered_error:
        return None, None

    repaired_program = str(program)
    changed = False

    if "lcm(" in repaired_program:
        repaired_program = _insert_import_after_sympy_wildcard(repaired_program, "from math import lcm")
        changed = True
    if "gcd(" in repaired_program:
        repaired_program = _insert_import_after_sympy_wildcard(repaired_program, "from math import gcd")
        changed = True

    if not changed:
        return None, None

    return repaired_program, "Switched integer gcd/lcm calls to math.* helpers so plain integers execute reliably."


def _repair_missing_print_output(program, error_message):
    lowered_error = str(error_message or "").lower()
    if "produced no output" not in lowered_error:
        return None, None

    lines = [line.rstrip() for line in str(program).splitlines()]
    if any(re.match(r"^\s*print\s*\(", line) for line in lines):
        return None, None

    preferred_names = ("ans", "answer", "final_answer", "result", "value", "solution")
    assigned_names = []
    for line in lines:
        match = re.match(r"^\s*([A-Za-z_]\w*)\s*=", line)
        if match:
            assigned_names.append(match.group(1))

    chosen_name = None
    for name in reversed(assigned_names):
        if name in preferred_names:
            chosen_name = name
            break

    if not chosen_name:
        return None, None

    repaired_lines = list(lines)
    repaired_lines.append(f"print({chosen_name})")
    return "\n".join(repaired_lines).strip(), f"Appended print({chosen_name}) so the derived final answer is emitted."


def _repair_sympy_coeff_function(program, error_message):
    lowered_error = str(error_message or "").lower()
    if "name 'coeff' is not defined" not in lowered_error:
        return None, None

    text = str(program or "")
    # Rewrite coeff(expr, sym, n) -> expr.coeff(sym, n)
    pattern = re.compile(r"(?<!\.)\bcoeff\s*\(\s*([^,]+?)\s*,\s*([^,]+?)\s*,\s*([^)]+?)\s*\)")
    if not pattern.search(text):
        return None, None

    # Expand first so coeff works on products/sums reliably.
    repaired = pattern.sub(r"expand(\1).coeff(\2, \3)", text)
    return repaired.strip(), "Rewrote SymPy coeff(expr, sym, n) calls to expand(expr).coeff(sym, n)."


def apply_local_python_repair(program, error_message):
    repaired_program, note = _repair_sympy_coeff_function(program, error_message)
    if repaired_program:
        return repaired_program, note

    repaired_program, note = _repair_sympy_solve_output_shape(program, error_message)
    if repaired_program:
        return repaired_program, note

    repaired_program, note = _repair_integer_lcm_or_gcd(program, error_message)
    if repaired_program:
        return repaired_program, note

    repaired_program, note = _repair_missing_print_output(program, error_message)
    if repaired_program:
        return repaired_program, note

    missing_name = _extract_missing_name(error_message)
    if not missing_name:
        return None, None

    import_line = NAME_ERROR_IMPORT_FIXES.get(missing_name)
    if import_line:
        repaired_program = _prepend_import_if_missing(program, import_line)
        return repaired_program, f"Added missing import for '{missing_name}'."

    if missing_name in SYMPY_NAME_FIXES:
        repaired_program = _prepend_import_if_missing(program, "from sympy import *")
        return repaired_program, f"Added missing SymPy import for '{missing_name}'."

    return None, None


def extract_missing_dependency(error_message):
    if not error_message:
        return None, None

    match = re.search(r"No module named ['\"]([^'\"]+)['\"]", str(error_message))
    if not match:
        return None, None

    module_name = match.group(1).strip()
    top_level_name = module_name.split(".", 1)[0]
    if module_name in AUTO_INSTALL_PACKAGE_MAP:
        return module_name, AUTO_INSTALL_PACKAGE_MAP[module_name]
    if top_level_name in AUTO_INSTALL_PACKAGE_MAP:
        return top_level_name, AUTO_INSTALL_PACKAGE_MAP[top_level_name]
    return None, None


def _default_dependency_installer(package_name):
    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
        package_name,
    ]
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=300,
    )
    output = "\n".join(part for part in [completed.stdout, completed.stderr] if part).strip()
    return completed.returncode == 0, output


def install_missing_dependency(error_message, installer=None):
    module_name, package_name = extract_missing_dependency(error_message)
    if not package_name:
        return {
            "module": None,
            "package": None,
            "installed": False,
            "message": None,
        }

    install_fn = installer or _default_dependency_installer
    installed, message = install_fn(package_name)
    return {
        "module": module_name,
        "package": package_name,
        "installed": bool(installed),
        "message": message,
    }


def execution_failure_reason(dataset, output, error_message):
    if error_message:
        return error_message

    if output is None or not str(output).strip():
        return "Program executed but produced no output. The code must print the derived final answer."

    return None


def should_attempt_program_repair(dataset, output, error_message):
    failure_reason = execution_failure_reason(dataset, output, error_message)
    if failure_reason is None:
        return False

    lowered = failure_reason.lower()
    return any(snippet in lowered for snippet in COMMON_REPAIR_ERROR_SNIPPETS)


def repair_program_until_runnable(program, dataset, execute_program, llm_repair=None, max_attempts=3):
    current_program = sanitize_generated_python(program)
    current_output, current_raw_error = execute_program(current_program)
    current_error = execution_failure_reason(dataset, current_output, current_raw_error)
    attempts = []
    seen_programs = {current_program} if current_program else set()

    if max_attempts is None:
        max_attempts = 0
    try:
        max_attempts = max(0, int(max_attempts))
    except (TypeError, ValueError):
        max_attempts = 0

    attempt_index = 0
    while attempt_index < max_attempts and should_attempt_program_repair(dataset, current_output, current_raw_error):
        attempt_index += 1
        trigger_error = execution_failure_reason(dataset, current_output, current_raw_error)

        repaired_program = None
        errors_fixed = None
        repair_source = None

        local_repaired_program, local_errors_fixed = apply_local_python_repair(current_program, trigger_error)
        local_repaired_program = sanitize_generated_python(local_repaired_program)
        if local_repaired_program and local_repaired_program != current_program and local_repaired_program not in seen_programs:
            repaired_program = local_repaired_program
            errors_fixed = local_errors_fixed
            repair_source = "local"
        elif llm_repair is not None:
            llm_repaired_program, llm_errors_fixed = llm_repair(current_program, trigger_error)
            llm_repaired_program = sanitize_generated_python(llm_repaired_program)
            if llm_repaired_program and llm_repaired_program != current_program and llm_repaired_program not in seen_programs:
                repaired_program = llm_repaired_program
                errors_fixed = llm_errors_fixed
                repair_source = "llm"

        if not repaired_program:
            break

        seen_programs.add(repaired_program)
        repaired_output, repaired_raw_error = execute_program(repaired_program)
        repaired_error = execution_failure_reason(dataset, repaired_output, repaired_raw_error)
        attempts.append(
            {
                "attempt": attempt_index,
                "source": repair_source,
                "trigger_error": trigger_error,
                "errors_fixed": errors_fixed,
                "repaired_program": repaired_program,
                "repaired_output": repaired_output,
                "repaired_error": repaired_error,
            }
        )

        current_program = repaired_program
        current_output = repaired_output
        current_raw_error = repaired_raw_error
        current_error = repaired_error

        if current_error is None:
            break

    return current_program, current_output, current_raw_error, current_error, attempts


def final_output_line(output):
    if output is None:
        return None

    lines = [line.strip() for line in str(output).splitlines() if line.strip()]
    if not lines:
        return None
    return lines[-1]

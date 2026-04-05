import re


_WORD_RE = re.compile(r"[A-Za-z]+")
_NUMBER_RE = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?(?:/\d+(?:,\d{3})*(?:\.\d+)?)?")
_EXPLICIT_COORDINATE_RE = re.compile(
    r"\(\s*-?(?:\d+(?:\.\d+)?|[A-Za-z][A-Za-z0-9_]*)\s*,\s*-?(?:\d+(?:\.\d+)?|[A-Za-z][A-Za-z0-9_]*)\s*\)"
)
_POINT_OBJECT_RE = re.compile(r"\b(?:Point|Point2D|Line|Segment|Circle|Polygon|Triangle)\s*\(")
_POINT_ASSIGNMENT_RE = re.compile(r"\b[A-Z]{1,2}\s*=\s*\(\s*[^,\n()]+\s*,\s*[^,\n()]+\s*\)")
_NUMERIC_TUPLE_RE = re.compile(r"\(\s*-?\d+(?:\.\d+)?\s*,\s*-?\d+(?:\.\d+)?\s*\)")

_IGNORED_TOKENS = {
    "absolute",
    "additional",
    "after",
    "again",
    "also",
    "answer",
    "arithmetic",
    "around",
    "before",
    "brief",
    "calculate",
    "carefully",
    "clear",
    "code",
    "comment",
    "comments",
    "compile",
    "concise",
    "condition",
    "conditions",
    "consider",
    "context",
    "current",
    "define",
    "derived",
    "detail",
    "details",
    "direct",
    "equation",
    "equations",
    "exact",
    "executable",
    "expression",
    "expressions",
    "explain",
    "final",
    "first",
    "form",
    "formula",
    "from",
    "given",
    "helpful",
    "ignore",
    "important",
    "include",
    "integer",
    "intermediate",
    "label",
    "line",
    "main",
    "match",
    "math",
    "mathematical",
    "mathematics",
    "method",
    "module",
    "modules",
    "needed",
    "numbers",
    "objects",
    "only",
    "option",
    "options",
    "output",
    "phrase",
    "print",
    "problem",
    "produce",
    "program",
    "proof",
    "python",
    "question",
    "reasoning",
    "regenerate",
    "relevant",
    "response",
    "result",
    "results",
    "scratch",
    "show",
    "single",
    "solve",
    "solved",
    "solver",
    "solution",
    "step",
    "steps",
    "story",
    "symbol",
    "symbols",
    "sympy",
    "terms",
    "than",
    "therefore",
    "these",
    "this",
    "those",
    "trace",
    "unrelated",
    "use",
    "using",
    "value",
    "values",
    "variable",
    "variables",
    "write",
}


def build_option_knowledge_demo_prompt(option_letters):
    return (
        "Read the current multiple-choice math question and produce only the background knowledge that helps solve it.\n"
        "Rules:\n"
        "- Focus only on the current question and its options.\n"
        "- Never reuse names, objects, numbers, equations, or answer choices from an earlier example.\n"
        "- Ignore any unrelated story details in surrounding context.\n"
        "- If the question is geometry, do not use hidden diagram coordinates or drawing positions; rely only on stated geometric facts.\n"
        f"- The valid answer letters for this dataset are {option_letters}.\n"
        "- Keep the knowledge short and directly useful.\n"
    )


def build_math_knowledge_demo_prompt():
    return (
        "Read the current math problem and produce only the background knowledge that helps solve it.\n"
        "Rules:\n"
        "- Focus only on the current question.\n"
        "- Never restate, summarize, or compare multiple different problems.\n"
        "- Never echo prompt examples, training examples, or earlier questions.\n"
        "- Ignore any unrelated surrounding context.\n"
        "- For geometry, do not use hidden diagram coordinates or drawing positions; rely only on stated geometric facts unless the problem explicitly gives coordinates.\n"
        "- Keep the knowledge short, concrete, and mathematically relevant.\n"
        "- Prefer 3 to 6 concise bullet points covering definitions, theorems, or setup ideas needed for this question only.\n"
    )


def build_option_wolfram_demo_prompt(option_letters):
    return (
        "Read the current multiple-choice math question and produce one Wolfram Alpha query that helps solve it.\n"
        "Rules:\n"
        "- Use only the numbers and relationships from the current question.\n"
        "- Never reuse names, objects, numbers, equations, or answer choices from an earlier example.\n"
        f"- The valid answer letters for this dataset are {option_letters}.\n"
        "- Prefer a direct arithmetic or algebra query when possible.\n"
        "- If the question is geometry, do not turn the figure into invented coordinates or use hidden diagram coordinates unless the problem explicitly gives coordinates.\n"
        "- For closest, nearest, approximate, or estimate questions, prefer computing the actual target quantity first so it can be compared against every option.\n"
        "- If the options are expressions, plan to evaluate the expressions rather than inventing your own rounding rule.\n"
        "Respond using exactly this format:\n"
        "Thought: <brief thought>\n"
        "Answer: <brief confirmation>\n"
        "Final Query: <single Wolfram Alpha query>\n"
    )


def build_option_program_demo_prompt(option_letters):
    return (
        "You are writing Python for one multiple-choice math question.\n"
        "Rules:\n"
        "- Solve ONLY the current question.\n"
        "- Never reuse names, objects, numbers, equations, comments, or variable names from any earlier example.\n"
        "- Use only the information present in the current question and options.\n"
        f"- The valid answer letters for this dataset are {option_letters}.\n"
        "- Prefer direct arithmetic or algebra grounded in the current numbers.\n"
        "- If the question is geometry, do not assign coordinates, extract coordinates from the diagram, or use analytic coordinate geometry unless the problem explicitly gives coordinates.\n"
        "- For closest, nearest, approximate, or estimate questions, compute the underlying target quantity and compare it against every option.\n"
        "- For closest, nearest, approximate, or estimate questions, do not round intermediate quantities unless the problem explicitly instructs that rounding.\n"
        "- For closest, nearest, approximate, or estimate questions, do not round first and compare later unless the problem explicitly tells you to round that way.\n"
        "- If the options use expressions or unusual roundings, evaluate the option expressions directly instead of assuming a default rounding method.\n"
        "- For closest, nearest, approximate, or estimate questions, print the computed target quantity and the numeric value of each option before printing the final letter.\n"
        "- Output only executable Python code.\n"
    )


def build_option_solution_demo_prompt(option_letters):
    return (
        "Given the current multiple-choice math question and any module context, write a concise solution.\n"
        "Rules:\n"
        "- Solve ONLY the current question.\n"
        "- Ignore any stray context that mentions people, objects, equations, or answer choices absent from the question.\n"
        "- Do not copy unrelated story details from earlier examples.\n"
        "- Keep the reasoning short and question-grounded.\n"
        "- If the question is geometry, do not use hidden diagram coordinates or invented coordinates; rely on stated geometric relationships unless coordinates are explicitly given.\n"
        "- You must reverse-check the final choice against the provided options before answering.\n"
        "- For closest, nearest, approximate, or estimate questions, compute the target quantity first, then compare every option numerically and choose the smallest difference.\n"
        "- If the options are expressions, evaluate those expressions before choosing the letter.\n"
        "- Do not assume a rounding rule unless the options clearly support it.\n"
        f'- End with exactly one final line in the form "The answer is [LETTER]" using one of {option_letters}.\n'
    )


def _content_tokens(text):
    tokens = set()
    for token in _WORD_RE.findall(str(text or "").lower().replace("_", " ")):
        if len(token) < 3:
            continue
        if token in _IGNORED_TOKENS:
            continue
        tokens.add(token)
    return tokens


def _number_tokens(text):
    values = set()
    for token in _NUMBER_RE.findall(str(text or "")):
        values.add(token.replace(",", ""))
    return values


def cross_problem_leak_details(question_text, candidate_text):
    question_tokens = _content_tokens(question_text)
    candidate_tokens = _content_tokens(candidate_text)
    question_numbers = _number_tokens(question_text)
    candidate_numbers = _number_tokens(candidate_text)
    return {
        "foreign_tokens": sorted(candidate_tokens - question_tokens),
        "overlap_tokens": sorted(candidate_tokens & question_tokens),
        "foreign_numbers": sorted(candidate_numbers - question_numbers),
        "overlap_numbers": sorted(candidate_numbers & question_numbers),
        "question_numbers": sorted(question_numbers),
        "candidate_numbers": sorted(candidate_numbers),
    }


def knowledge_leak_details(question_text, candidate_text):
    details = cross_problem_leak_details(question_text, candidate_text)
    text = str(candidate_text or "")
    details.update(
        {
            "numbered_headings": len(re.findall(r"(?m)^\s*\d+\.\s+.+$", text)),
            "problem_headings": len(re.findall(r"(?im)^\s*\d+\.\s+.*(?:problem|question)\b.*:", text)),
            "question_labels": len(re.findall(r"(?im)^\s*question\s*:", text)),
            "knowledge_labels": len(re.findall(r"(?im)^\s*(?:relevant concepts?|knowledge retrieval|knowledge)\s*:", text)),
            "separator_lines": len(re.findall(r"(?m)^\s*---+\s*$", text)),
        }
    )
    return details


def looks_like_knowledge_leak(question_text, candidate_text):
    details = knowledge_leak_details(question_text, candidate_text)
    foreign_tokens = details["foreign_tokens"]
    overlap_tokens = details["overlap_tokens"]
    foreign_numbers = details["foreign_numbers"]
    overlap_numbers = details["overlap_numbers"]
    question_numbers = details["question_numbers"]
    candidate_numbers = details["candidate_numbers"]

    if details["problem_headings"] >= 2:
        return True
    if details["question_labels"] >= 2:
        return True
    if details["knowledge_labels"] >= 2 and details["separator_lines"] >= 1:
        return True
    if details["numbered_headings"] >= 3 and details["separator_lines"] >= 2:
        return True
    if (
        len(question_numbers) >= 2
        and len(candidate_numbers) >= 3
        and not overlap_numbers
        and len(foreign_numbers) >= 3
        and details["numbered_headings"] >= 2
    ):
        return True
    if len(foreign_tokens) >= 10 and len(overlap_tokens) <= 1 and (
        details["knowledge_labels"] >= 2 or details["numbered_headings"] >= 2
    ):
        return True
    return False


def looks_like_cross_problem_leak(question_text, candidate_text, *, mode="program"):
    details = cross_problem_leak_details(question_text, candidate_text)
    foreign_tokens = details["foreign_tokens"]
    overlap_tokens = details["overlap_tokens"]
    foreign_numbers = details["foreign_numbers"]
    overlap_numbers = details["overlap_numbers"]
    question_numbers = details["question_numbers"]
    candidate_numbers = details["candidate_numbers"]

    if mode == "program":
        if len(question_numbers) >= 2 and len(candidate_numbers) >= 2 and not overlap_numbers and len(foreign_numbers) >= 2:
            return True
        if len(foreign_tokens) >= 4 and len(overlap_tokens) <= 1:
            return True
        return False

    if len(question_numbers) >= 2 and len(candidate_numbers) >= 3 and not overlap_numbers and len(foreign_numbers) >= 2:
        return True
    if len(foreign_tokens) >= 5 and len(overlap_tokens) <= 1:
        return True
    return False


def question_explicitly_uses_coordinates(question_text):
    cleaned = str(question_text or "")
    lowered = cleaned.lower()
    coordinate_markers = (
        "coordinate",
        "coordinates",
        "coordinate plane",
        "cartesian",
        "x-coordinate",
        "y-coordinate",
        "xy-plane",
    )
    return any(marker in lowered for marker in coordinate_markers) or bool(_EXPLICIT_COORDINATE_RE.search(cleaned))


def program_uses_hidden_geometry_coordinates(question_text, candidate_program):
    if question_explicitly_uses_coordinates(question_text):
        return False

    program_text = str(candidate_program or "")
    if not program_text.strip():
        return False

    return bool(
        _POINT_OBJECT_RE.search(program_text)
        or _POINT_ASSIGNMENT_RE.search(program_text)
        or _NUMERIC_TUPLE_RE.search(program_text)
    )

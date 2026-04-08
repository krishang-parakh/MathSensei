import re


_WORD_RE = re.compile(r"[A-Za-z]+")

_IGNORED_GSM_TOKENS = {
    "answer",
    "arithmetic",
    "begin",
    "calculate",
    "clear",
    "code",
    "comment",
    "comments",
    "compile",
    "concise",
    "context",
    "current",
    "define",
    "derived",
    "direct",
    "equation",
    "equations",
    "exact",
    "explain",
    "final",
    "first",
    "from",
    "helper",
    "helpers",
    "ignore",
    "integer",
    "line",
    "math",
    "module",
    "modules",
    "needed",
    "only",
    "output",
    "print",
    "problem",
    "program",
    "python",
    "question",
    "result",
    "results",
    "short",
    "solve",
    "solved",
    "solver",
    "step",
    "steps",
    "story",
    "symbol",
    "symbols",
    "sympy",
    "these",
    "this",
    "using",
    "use",
    "useful",
    "value",
    "variable",
    "variables",
    "write",
}


def build_gsm_knowledge_demo_prompt():
    return (
        "Read the current GSM-style word problem and generate only the background knowledge that is helpful for that "
        "question.\n"
        "Rules:\n"
        "- Focus only on the current question.\n"
        "- Never reuse names, objects, numbers, or equations from an earlier example.\n"
        "- If any surrounding context mentions unrelated story details, ignore it.\n"
        "- Keep the knowledge short, concrete, and arithmetic-focused.\n"
    )


def build_gsm_wolfram_demo_prompt():
    return (
        "Read the current GSM-style word problem and produce one Wolfram Alpha query that directly helps solve it.\n"
        "Rules:\n"
        "- Use only the numbers and story details from the current question.\n"
        "- Never reuse names, objects, or equations from an earlier example.\n"
        "- Prefer a direct arithmetic query when possible.\n"
        "- For rate, money, or unit problems, convert the setup into plain arithmetic with numbers and operators only.\n"
        "- Do not include unit words like eggs/day, days, dollars, per dozen, miles, or hours inside the final query.\n"
        "- Use explicit parentheses for multi-step arithmetic so the intended order of operations is unambiguous.\n"
        "- The final query should evaluate directly to the requested numeric quantity.\n"
        "- If a variable is needed, use a generic variable like x rather than a copied story-specific name.\n"
        "- AVOID generating vague or empty queries (e.g., a bare number with no operation, or natural language).\n"
        "- AVOID repeating the same query if it failed to produce useful results.\n"
        "- Use mathematical operators (+, -, *, /, ^) and functions explicitly.\n"
        "Respond using exactly this format:\n"
        "Thought: <brief thought about the arithmetic>\n"
        "Answer: <brief confirmation of the calculation>\n"
        "Final Query: <single concrete Wolfram Alpha query with all operators written out>\n"
    )


def build_gsm_program_demo_prompt():
    return (
        "You are writing Python for one GSM-style arithmetic word problem.\n"
        "Rules:\n"
        "- Solve ONLY the current question.\n"
        "- Never reuse names, objects, numbers, equations, comments, or variable names from any earlier example.\n"
        "- If prior context mentions unrelated story details, ignore it.\n"
        "- Prefer direct arithmetic with variable names grounded in the current question.\n"
        "- Use SymPy only when a short equation is genuinely needed.\n"
        "- Keep comments crisp and tied to the current question.\n"
        "- Output only executable Python code.\n"
    )


def build_gsm_solution_demo_prompt():
    return (
        "Given the current GSM-style word problem and any module context, write a concise solution.\n"
        "Rules:\n"
        "- Solve ONLY the current question.\n"
        "- Ignore any stray context that mentions people, objects, or equations absent from the question.\n"
        "- Do not copy unrelated story details from earlier examples.\n"
        "- Keep the reasoning short and question-grounded.\n"
        '- End with exactly one final line in the form "Final Answer: [integer]".\n'
        "- The final answer must use only digits with an optional leading minus sign.\n"
    )


def _content_tokens(text):
    tokens = set()
    for token in _WORD_RE.findall(str(text or "").lower().replace("_", " ")):
        if len(token) < 3:
            continue
        if token in _IGNORED_GSM_TOKENS:
            continue
        tokens.add(token)
    return tokens


def gsm_story_leak_tokens(question_text, candidate_text):
    question_tokens = _content_tokens(question_text)
    candidate_tokens = _content_tokens(candidate_text)
    return sorted(candidate_tokens - question_tokens)


def looks_like_gsm_story_leak(question_text, candidate_text):
    candidate_tokens = _content_tokens(candidate_text)
    if len(candidate_tokens) < 4:
        return False

    question_tokens = _content_tokens(question_text)
    overlap = candidate_tokens & question_tokens
    foreign_tokens = candidate_tokens - question_tokens
    return len(overlap) == 0 and len(foreign_tokens) >= 3

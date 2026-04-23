import ast
import json
import re
from typing import Callable, Optional, Tuple

from core.question_isolation import looks_like_knowledge_leak


def remove_wrapping_backticks(input_str: Optional[str]) -> Optional[str]:
    if input_str is None:
        return None
    if input_str.startswith("`") and input_str.endswith("`"):
        return input_str[1:-1]
    return input_str


def build_wolfram_answer_cleaner_prompt(query: str, result) -> str:
    result_str = str(result)
    return (
        f"I called Wolfram alpha API using {query} and it gave me this answer as a dictionary object.\n "
        f"{result_str}\n.Can you get the answer for me from this object?"
    )


def parse_wolfram_query_response(text: Optional[str]):
    cleaned = str(text or "").strip()
    if not cleaned or "Final Query:" not in cleaned:
        return None

    final_idx = cleaned.find("Final Query:")
    answer_idx = cleaned.find("Answer:")
    thought_block = cleaned[:answer_idx] if answer_idx != -1 else cleaned[:final_idx]
    query_block = cleaned[final_idx + len("Final Query:"):]
    first_line = query_block.split("\n")[0].strip()
    if "##" in first_line:
        first_line = first_line[:first_line.index("##")].strip()
    query = remove_wrapping_backticks(first_line)
    if not query:
        return None

    thought = re.sub(r"^\s*Thought\s*:\s*", "", thought_block.strip(), flags=re.IGNORECASE)
    return {
        "thought": thought or "",
        "query": query,
    }


def format_wolfram_trace(trace) -> str:
    lines = []
    for index, step in enumerate(trace or [], start=1):
        if not isinstance(step, dict):
            continue
        thought = str(step.get("thought") or "").strip()
        query = str(step.get("query") or "").strip()
        output = str(step.get("output") or "").strip()
        error = str(step.get("error") or "").strip()
        source = str(step.get("source") or "").strip()

        lines.append(f"Step {index}:")
        if thought:
            lines.append(f"Thought: {thought}")
        if query:
            lines.append(f"Query: {query}")
        if output:
            lines.append(f"Output: {output}")
        if error:
            lines.append(f"Error: {error}")
        if source:
            lines.append(f"Source: {source}")
        lines.append("")
    return "\n".join(lines).strip()


def normalize_wolfram_query_signature(text: Optional[str]) -> str:
    cleaned = remove_wrapping_backticks(str(text or "").strip())
    if not cleaned:
        return ""
    cleaned = cleaned.replace("−", "-")
    cleaned = re.sub(r"\s+", "", cleaned)
    return cleaned.lower()


def normalize_wolfram_output_signature(text: Optional[str]) -> str:
    cleaned = remove_wrapping_backticks(str(text or "").strip())
    if not cleaned:
        return ""
    cleaned = cleaned.replace("−", "-")
    cleaned = cleaned.replace("≈", "=")
    cleaned = re.sub(r"\s+", "", cleaned)
    cleaned = re.sub(r"[`'\"]", "", cleaned)
    return cleaned.lower()


def detect_wolfram_loop_reason(trace, next_query: Optional[str] = None) -> Optional[str]:
    trace = [step for step in (trace or []) if isinstance(step, dict)]
    if not trace:
        return None

    normalized_queries = [normalize_wolfram_query_signature(step.get("query")) for step in trace]
    normalized_outputs = [normalize_wolfram_output_signature(step.get("output")) for step in trace]

    if next_query:
        next_query_signature = normalize_wolfram_query_signature(next_query)
        if next_query_signature and next_query_signature in {signature for signature in normalized_queries if signature}:
            return "Wolfram planner repeated a previous query before reaching the requested final quantity"

    if len(trace) < 2:
        return None

    latest_query = normalized_queries[-1]
    previous_query = normalized_queries[-2]
    latest_output = normalized_outputs[-1]
    previous_output = normalized_outputs[-2]

    if latest_query and latest_output and latest_query == previous_query and latest_output == previous_output:
        return "Wolfram repeated the same query and output without making progress toward the requested final quantity"

    if latest_output and latest_output == previous_output:
        return "Wolfram produced the same output again without making progress toward the requested final quantity"

    return None


def build_wolfram_next_step_prompt(question_text: str, trace) -> str:
    trace_text = format_wolfram_trace(trace) or "(no Wolfram steps yet)"
    return (
        "You are supervising a Wolfram Alpha tool for a math problem.\n"
        "Decide whether the Wolfram steps so far fully answer the original question.\n"
        "Important rules:\n"
        "- Do not finalize if the current output only solves an intermediate subproblem.\n"
        "- Finalize only when every condition in the original question has been satisfied.\n"
        "- If the original question includes answer choices, the final answer must align with those choices rather than stopping at an unmatched raw number.\n"
        "- If you are checking answer choices one by one, do not finalize until every choice has been explicitly checked or ruled out.\n"
        "- For closest, nearest, approximate, or estimate questions, compare the computed quantity against every numeric option and choose the closest valid option.\n"
        "- For closest, nearest, approximate, or estimate questions, explicitly evaluate every option and compare absolute differences before naming an option. Never guess an option letter.\n"
        "- If the options are expressions, evaluate the expressions before choosing the final option.\n"
        "- If more work is needed, give exactly one next Wolfram Alpha query.\n"
        "- Never suggest a query that repeats an earlier query with only cosmetic formatting changes.\n"
        "- If the latest Wolfram output repeats an earlier output with no new progress, do not keep cycling.\n"
        "- Reuse earlier Wolfram results when possible.\n"
        "- Prefer continuing with Wolfram queries over solving the rest yourself unless the final answer is already fully determined by the trace.\n"
        "- For arithmetic, rate, money, or unit word problems, finalize only when the Wolfram output is the plain computed numeric quantity asked for.\n"
        "- If the latest Wolfram output looks like an input interpretation, leftover unit expression, or otherwise not the final computed quantity, continue with a cleaner numeric query using explicit parentheses and no unit words.\n"
        "- Keep the final answer concise and answer-only.\n\n"
        f"Question:\n{question_text.strip()}\n\n"
        f"Wolfram steps so far:\n{trace_text}\n\n"
        "Respond using exactly this format:\n"
        "Status: CONTINUE or FINAL\n"
        "Thought: <brief reason>\n"
        "Next Query: <single Wolfram Alpha query>\n"
        "Final Answer: <answer only>\n\n"
        "Rules for the fields:\n"
        "- If Status is CONTINUE, fill Next Query and leave Final Answer blank.\n"
        "- If Status is FINAL, fill Final Answer and leave Next Query blank.\n"
        "- Never output both a next query and a final answer.\n"
    )


def _extract_labeled_block(text: str, label: str):
    pattern = rf"(?ims)^\s*{re.escape(label)}\s*:\s*(.*?)(?=^\s*(?:Status|Thought|Next Query|Final Answer)\s*:|\Z)"
    match = re.search(pattern, text)
    if not match:
        return None
    value = match.group(1).strip()
    return remove_wrapping_backticks(value) if value else None


def parse_wolfram_next_step_response(text: Optional[str]):
    cleaned = str(text or "").strip()
    if not cleaned:
        return None

    for loader in (json.loads, ast.literal_eval):
        try:
            parsed = loader(cleaned)
        except Exception:
            parsed = None
        if isinstance(parsed, dict):
            status = str(parsed.get("status") or "").strip().upper()
            if status in {"DONE", "COMPLETE"}:
                status = "FINAL"
            if status in {"FINAL", "CONTINUE"}:
                next_query = remove_wrapping_backticks(str(parsed.get("next_query") or "").strip()) or None
                final_answer = remove_wrapping_backticks(str(parsed.get("final_answer") or "").strip()) or None
                thought = str(parsed.get("thought") or "").strip()
                if status == "FINAL" and final_answer:
                    return {"status": status, "thought": thought, "next_query": None, "final_answer": final_answer}
                if status == "CONTINUE" and next_query:
                    return {"status": status, "thought": thought, "next_query": next_query, "final_answer": None}

    status_match = re.search(r"(?im)^\s*Status\s*:\s*([A-Za-z_ -]+)\s*$", cleaned)
    if not status_match:
        return None

    status = status_match.group(1).strip().upper().replace(" ", "_")
    if status in {"DONE", "COMPLETE"}:
        status = "FINAL"
    if status not in {"FINAL", "CONTINUE"}:
        return None

    thought = _extract_labeled_block(cleaned, "Thought") or ""
    next_query = _extract_labeled_block(cleaned, "Next Query")
    final_answer = _extract_labeled_block(cleaned, "Final Answer")

    if status == "FINAL" and final_answer:
        return {"status": status, "thought": thought, "next_query": None, "final_answer": final_answer}
    if status == "CONTINUE" and next_query:
        return {"status": status, "thought": thought, "next_query": next_query, "final_answer": None}
    return None


def resolve_wolfram_answer(
    query: str,
    result,
    *,
    chat_callable: Callable[[str, int], str],
    max_tokens: int = 5000,
    wolfram_model: str = "no",
    text_completion_callable: Optional[Callable[[str, int], str]] = None,
    gemini_callable: Optional[Callable[[str], str]] = None,
) -> str:
    prompt = build_wolfram_answer_cleaner_prompt(query, result)

    if wolfram_model in {"text_davinci_002", "text_davinci_003"} and text_completion_callable is not None:
        return text_completion_callable(prompt, max_tokens)

    if wolfram_model == "gemini" and gemini_callable is not None:
        return gemini_callable(prompt)

    return chat_callable(prompt, max_tokens)


def build_knowledge_retrieval_prompt(
    demo_prompt: str,
    query_text: str,
    context: str = "",
    *,
    mode: str = "question",
) -> Tuple[str, str]:
    if mode == "query":
        test_prompt = f"Query:\n{query_text}\nKnowledge:\n"
        full_prompt = demo_prompt + "\n" + test_prompt
        return test_prompt, full_prompt

    if context:
        test_prompt = f"Question: {query_text}\n\n{context}\n\nKnowledge Retrieval:\n"
    else:
        test_prompt = f"Question: {query_text}\n\nKnowledge Retrieval:\n"

    full_prompt = demo_prompt + "\n\n" + test_prompt
    return test_prompt, full_prompt


STATIC_MODEL_PIPELINES = {
    "cot": ["solution_generator"],
    "sg": ["solution_generator"],
    "kr_sg": ["knowledge_retrieval", "solution_generator"],
    "kr_walpha_sg": ["knowledge_retrieval", "wolfram_alpha_search", "solution_generator"],
    "kr_pg_walpha_sg": ["knowledge_retrieval", "program_generator", "program_executor", "wolfram_alpha_search", "solution_generator"],
    "walpha_sg": ["wolfram_alpha_search", "solution_generator"],
    "walpha_pg_sg": ["wolfram_alpha_search", "program_generator", "program_executor", "solution_generator"],
    "pg_walpha_sg": ["program_generator", "program_executor", "wolfram_alpha_search", "solution_generator"],
    "bing_sg": ["bing_search", "solution_generator"],
    "bing_pg_sg": ["bing_search", "program_generator", "program_executor", "solution_generator"],
    "pg_bing_sg": ["program_generator", "program_executor", "bing_search", "solution_generator"],
    "bing_walpha_sg": ["bing_search", "wolfram_alpha_search", "solution_generator"],
    "walpha_bing_sg": ["wolfram_alpha_search", "bing_search", "solution_generator"],
    "bing_pg_walpha_sg": ["bing_search", "program_generator", "program_executor", "wolfram_alpha_search", "solution_generator"],
}


def normalize_module_sequence(modules) -> list:
    normalized = []
    seen = set()
    for module in modules or []:
        cleaned = str(module).strip() if module is not None else ""
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        normalized.append(cleaned)
    return normalized


def extract_example_metadata(example) -> dict:
    example = example or {}

    def _clean(value):
        if value in (None, ""):
            return ""
        return str(value).strip()

    topic = _clean(
        example.get("type")
        or example.get("subject")
        or example.get("topic")
        or example.get("category")
    )
    level = _clean(
        example.get("level")
        or example.get("difficulty")
        or example.get("grade")
    )
    return {
        "topic": topic,
        "level": level,
    }


def build_initial_question_response(example, dataset) -> str:
    example = example or {}
    if str(dataset).upper() != "MATH":
        return ""

    metadata = extract_example_metadata(example)
    problem_type = metadata.get("topic")
    level = metadata.get("level")
    lines = []
    if problem_type not in (None, ""):
        lines.append(f"Mathematics Problem Type:{problem_type}")
    if level not in (None, ""):
        lines.append(f"Level of Problem:{level}")
    return "\n".join(lines).strip()


def build_knowledge_response_context(question_text, knowledge) -> str:
    if knowledge in (None, ""):
        return ""
    if looks_like_knowledge_leak(question_text, knowledge):
        return ""
    return f"Knowledge Retrieval:\n{knowledge}".strip()


def build_wolfram_response_context(trace, final_answer) -> str:
    lines = []
    for step in trace or []:
        if not isinstance(step, dict):
            continue
        step_number = step.get("step")
        thought = step.get("thought")
        query = step.get("query")
        output = step.get("output")
        if thought:
            lines.append(f"Wolfram Step {step_number} Thought: {thought}")
        if query:
            lines.append(f"Wolfram Step {step_number} Query: {query}")
        if output:
            lines.append(f"Wolfram Step {step_number} Output: {output}")

    if final_answer not in (None, ""):
        lines.append(f"Wolfram Final Answer: {final_answer}")

    return "\n".join(lines).strip()


def rebuild_cached_response(cache_row, *, modules=None) -> str:
    cache_row = cache_row or {}
    example = cache_row.get("example") or {}
    question_text = (
        example.get("problem")
        or example.get("question")
        or example.get("Question")
        or ""
    )
    selected_modules = normalize_module_sequence(modules)
    if not selected_modules:
        selected_modules = [
            "knowledge_retrieval",
            "bing_search",
            "wolfram_alpha_search",
            "program_executor",
            "solution_generator",
        ]

    sections = []
    initial_response = build_initial_question_response(cache_row.get("example"), cache_row.get("dataset"))
    if initial_response:
        sections.append(initial_response)

    for module_name in selected_modules:
        if module_name == "knowledge_retrieval":
            knowledge = cache_row.get("knowledge_retrieval:output")
            knowledge_context = build_knowledge_response_context(question_text, knowledge)
            if knowledge_context:
                sections.append(knowledge_context)
        elif module_name == "bing_search":
            bing_output = cache_row.get("bing_search:output")
            if bing_output not in (None, ""):
                sections.append(f"Bing search response:\n{bing_output}")
        elif module_name == "wolfram_alpha_search":
            wolfram_context = build_wolfram_response_context(
                cache_row.get("wolfram_alpha_search:trace") or [],
                cache_row.get("wolfram_alpha_search:output"),
            )
            if wolfram_context:
                sections.append(wolfram_context)
        elif module_name == "program_executor":
            program = cache_row.get("program")
            output = cache_row.get("program_executor:output")
            error = cache_row.get("program_executor:error")
            if program not in (None, "") and output not in (None, ""):
                sections.append(f"Python generator:\n{program}\n\nPython output:\n{output}")
            elif program not in (None, "") and error not in (None, ""):
                sections.append(f"Python generator:\n{program}\n\nPython execution error:\n{error}")
        elif module_name == "solution_generator":
            solution = cache_row.get("solution_generator:output") or cache_row.get("solution")
            if solution not in (None, ""):
                sections.append(f"Solution:\n{solution}")

    return "\n\n".join(section.strip() for section in sections if str(section).strip()).strip()


def resolve_modules_for_model(
    model_name: str,
    *,
    refine: str = "no",
    custom_modules=None,
    planner_callable: Optional[Callable[[], list]] = None,
) -> list:
    if custom_modules is not None:
        return normalize_module_sequence(custom_modules)

    if model_name == "planner":
        if planner_callable is None:
            return ["solution_generator"]
        return normalize_module_sequence(planner_callable())

    refine_enabled = str(refine).lower() != "no"
    if model_name in {"pg_sg", "pot"}:
        if refine_enabled:
            return ["python_generator_refine_executor", "solution_generator"]
        return ["program_generator", "program_executor", "solution_generator"]

    if model_name == "kr_pg_sg":
        if refine_enabled:
            return ["knowledge_retrieval", "python_generator_refine_executor", "solution_generator"]
        return ["knowledge_retrieval", "program_generator", "program_executor", "solution_generator"]

    if model_name == "kr_pg_walpha_sg":
        if refine_enabled:
            return ["knowledge_retrieval", "python_generator_refine_executor", "wolfram_alpha_search", "solution_generator"]
        return ["knowledge_retrieval", "program_generator", "program_executor", "wolfram_alpha_search", "solution_generator"]

    modules = STATIC_MODEL_PIPELINES.get(model_name)
    if modules is None:
        raise ValueError(f"Unsupported model '{model_name}'")
    return normalize_module_sequence(modules)

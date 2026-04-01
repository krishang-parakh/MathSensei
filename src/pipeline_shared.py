from typing import Callable, Optional, Tuple


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


def resolve_wolfram_answer(
    query: str,
    result,
    *,
    chat_callable: Callable[[str, int], str],
    max_tokens: int = 5000,
    wolfram_model: str = "no",
    text_davinci003_callable: Optional[Callable[[str, float, int], str]] = None,
    gemini_callable: Optional[Callable[[str], str]] = None,
) -> str:
    prompt = build_wolfram_answer_cleaner_prompt(query, result)

    if wolfram_model == "text_davinci_003" and text_davinci003_callable is not None:
        return text_davinci003_callable(prompt, 0.5, max_tokens)

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

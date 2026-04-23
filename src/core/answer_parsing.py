import re


def clean_answer_text(value):
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _last_group_match(text, patterns):
    for pattern in patterns:
        matches = list(re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE))
        if matches:
            candidate = matches[0].group(1).strip().strip(".")
            if candidate:
                return candidate
    return None


def extract_boxed_answer(text):
    cleaned = clean_answer_text(text)
    if not cleaned:
        return None

    marker = "\\boxed{"
    start = cleaned.rfind(marker)
    if start == -1:
        return None

    idx = start + len(marker)
    depth = 1
    collected = []

    while idx < len(cleaned):
        char = cleaned[idx]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                candidate = "".join(collected).strip()
                return candidate or None
        collected.append(char)
        idx += 1

    return None


def extract_tagged_answer(text):
    cleaned = clean_answer_text(text)
    if not cleaned:
        return None

    patterns = [
        r"####\s*([^\n]+)",
        r"\bObservation\s+\d+\s*:\s*([^\n]+)",
        r"\bFinal Answer\s*:\s*([^\n]+)",
        r"\bThe correct answer is\s*([^\n]+)",
        r"\bThe answer is\s*([^\n]+)",
        r"(?<![A-Za-z])Answer\s*:\s*([^\n]+)",
    ]
    return _last_group_match(cleaned, patterns)


def extract_option_letter(text, allowed="ABCDE"):
    cleaned = clean_answer_text(text)
    if not cleaned:
        return None

    escaped_allowed = re.escape(str(allowed))
    patterns = [
        rf"\bFinal Answer\s*:\s*([{escaped_allowed}])\b",
        rf"\bThe correct answer is\s*([{escaped_allowed}])\b",
        rf"\bThe answer is\s*([{escaped_allowed}])\b",
        rf"\bAnswer\s*[: ]\s*([{escaped_allowed}])\b",
        rf"\bOption\s*([{escaped_allowed}])\b",
        rf"(?:^|\n)\s*\(?([{escaped_allowed}])\)?[\)\.:]\s*",
        rf"(?:^|\n)\s*([{escaped_allowed}])\s*$",
    ]
    match = _last_group_match(cleaned, patterns)
    return match.upper() if match else None


def extract_final_answer_option_letter(text, allowed="ABCDE"):
    cleaned = clean_answer_text(text)
    if not cleaned:
        return None

    escaped_allowed = re.escape(str(allowed))
    patterns = [
        rf"\bFinal Answer\s*:\s*([{escaped_allowed}])\b",
        rf"####\s*([{escaped_allowed}])\b",
    ]
    match = _last_group_match(cleaned, patterns)
    return match.upper() if match else None


def option_listing_line(text, allowed="ABCDE"):
    cleaned = clean_answer_text(text)
    if not cleaned:
        return False

    stripped = cleaned.replace("**", "").replace("__", "").replace("`", "").strip()
    escaped_allowed = re.escape(str(allowed))
    return bool(
        re.match(
            rf"^(?:option\s+)?[{escaped_allowed}][\)\.\:]\s*",
            stripped,
            flags=re.IGNORECASE,
        )
    )


def candidate_looks_like_option_listing(text, allowed="ABCDE"):
    if not option_listing_line(text, allowed=allowed):
        return False

    cleaned = clean_answer_text(text)
    if not cleaned:
        return False

    lowered = cleaned.lower()
    if any(
        marker in lowered
        for marker in (
            "final answer",
            "the correct answer is",
            "the answer is",
            "answer:",
            "correct option",
            "correct answer",
            " is correct",
            "matches",
            "matched",
            "not correct",
            "incorrect",
            "wrong",
            "therefore",
            "thus",
            "hence",
        )
    ):
        return False

    return True


def text_has_option_listing_block(text, allowed="ABCDE", threshold=2):
    cleaned = clean_answer_text(text)
    if not cleaned:
        return False

    count = 0
    for line in cleaned.splitlines():
        if option_listing_line(line, allowed=allowed):
            count += 1
            if count >= threshold:
                return True

    return False


def candidate_has_rejection_cue(text):
    cleaned = clean_answer_text(text)
    if not cleaned:
        return False

    normalized = cleaned.replace("**", "").replace("__", "").replace("`", "").strip()
    patterns = [
        r"\bnot\s+correct\b",
        r"\bincorrect\b",
        r"\bwrong\b",
        r"\bnot\s+the\s+answer\b",
        r"\bdoes\s+not\s+match\b",
        r"\bdoesn't\s+match\b",
        r"\bnot\s+match(?:ing)?\b",
        r"\bdoes\s+not\s+satisfy\b",
        r"\bdoesn't\s+satisfy\b",
    ]
    return any(re.search(pattern, normalized, flags=re.IGNORECASE) for pattern in patterns)


_NUMBER_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}

_SCALE_WORDS = {
    "hundred": 100,
    "thousand": 1000,
    "million": 1_000_000,
    "billion": 1_000_000_000,
}


def _clean_candidate_text(value):
    cleaned = clean_answer_text(value)
    if not cleaned:
        return None

    cleaned = re.sub(r"^[\*\u2022]+\s*", "", cleaned)
    cleaned = re.sub(r"^-\s+(?=[^\d.])", "", cleaned)
    cleaned = cleaned.strip().strip("`").strip("*_").strip()
    cleaned = cleaned.rstrip(".,;")
    cleaned = cleaned.rstrip("*_").strip()
    return cleaned or None


def number_words_to_numeric_string(text):
    cleaned = _clean_candidate_text(text)
    if not cleaned:
        return None

    normalized = cleaned.lower().replace("-", " ")
    normalized = re.sub(r"\b(?:exactly|approximately|about)\b", " ", normalized)
    normalized = re.sub(r"[,_]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    if not normalized:
        return None

    tokens = normalized.split()
    allowed = set(_NUMBER_WORDS) | set(_SCALE_WORDS) | {"and", "minus", "negative", "point"}
    if any(token not in allowed for token in tokens):
        return None

    sign = 1
    if tokens and tokens[0] in {"minus", "negative"}:
        sign = -1
        tokens = tokens[1:]
    if not tokens:
        return None

    total = 0
    current = 0
    decimal_digits = []
    decimal_mode = False
    saw_number = False

    for token in tokens:
        if token == "and":
            continue
        if token == "point":
            if decimal_mode or not saw_number:
                return None
            decimal_mode = True
            continue
        if decimal_mode:
            digit = _NUMBER_WORDS.get(token)
            if digit is None or digit >= 10:
                return None
            decimal_digits.append(str(digit))
            continue
        if token in _NUMBER_WORDS:
            current += _NUMBER_WORDS[token]
            saw_number = True
            continue
        if token == "hundred":
            current = max(1, current) * 100
            saw_number = True
            continue
        scale = _SCALE_WORDS.get(token)
        if scale is None:
            return None
        total += max(1, current) * scale
        current = 0
        saw_number = True

    if not saw_number:
        return None

    integer_part = str(sign * (total + current))
    if not decimal_digits:
        return integer_part

    prefix = "-" if integer_part.startswith("-") else ""
    integer_digits = integer_part[1:] if prefix else integer_part
    return f"{prefix}{integer_digits}.{''.join(decimal_digits)}"


def _looks_like_label_fragment(text):
    return bool(re.search(r"[A-Za-z\\_()]", text or ""))


def _extract_rhs_candidate(line):
    cleaned = _clean_candidate_text(line)
    if not cleaned:
        return None

    parts = re.split(r"\s*(?:->|=>|→|=|:)\s*", cleaned)
    if len(parts) < 2:
        return None

    left = parts[-2].strip()
    right = parts[-1].strip()
    if not right or not _looks_like_label_fragment(left):
        return None
    concise_right = _clean_candidate_text(right)
    if not concise_right:
        return None
    if option_listing_line(cleaned):
        lowered_right = concise_right.lower()
        if not any(
            marker in lowered_right
            for marker in (
                "answer",
                "correct",
                "match",
                "matches",
                "matched",
                "solution",
                "therefore",
                "thus",
                "hence",
            )
        ):
            return None
    if len(concise_right) > 72:
        return None
    if re.search(r",\s*(?:and|but|or|which|that|because|since|while|where|when)\b", concise_right, re.IGNORECASE):
        return None
    if len(re.findall(r"[A-Za-z]+", concise_right)) > 4:
        return None
    return concise_right


def _extract_verbal_tail_candidate(line):
    cleaned = _clean_candidate_text(line)
    if not cleaned:
        return None

    patterns = [
        r"\b(?:final answer|answer|result|output|value|solution|probability)\b[^.\n]{0,80}?\b(?:is|are|was|were|equals?)\s+(.+)$",
        r"\b(?:therefore|thus|hence|so)\b[^.\n]{0,80}?\b(?:is|are|was|were|equals?)\s+(.+)$",
    ]
    return _last_group_match(cleaned, patterns)


def extract_answer_candidates(text, max_lines=5):
    cleaned = clean_answer_text(text)
    if not cleaned:
        return []

    candidates = []

    def add(value):
        candidate = _clean_candidate_text(value)
        if candidate and candidate not in candidates:
            candidates.append(candidate)

    add(extract_tagged_answer(cleaned))
    add(extract_boxed_answer(cleaned))

    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]

    # Most natural-language solutions place the answer at the end, but tool outputs (Python/WA)
    # often print intermediate diagnostics after the true result. Prefer answer-looking lines
    # within the tail window first so we don't accidentally capture a trailing debug number.
    window = lines[-max_lines:] if lines else [cleaned]
    answerish = []
    answerish_seen = set()
    answerish_re = re.compile(
        r"\b(final answer|answer|result|output|coefficient|probability|solution)\b",
        flags=re.IGNORECASE,
    )
    for line in window:
        if line and answerish_re.search(line) and line not in answerish_seen:
            answerish.append(line)
            answerish_seen.add(line)

    # Keep the historical bias towards the last few lines, but only after the answer-ish lines.
    tail_lines = answerish + [line for line in reversed(window) if line not in answerish_seen]

    for line in tail_lines:
        add(extract_tagged_answer(line))
        add(_extract_verbal_tail_candidate(line))
        add(_extract_rhs_candidate(line))
        add(line)

    if len(lines) <= 1:
        add(cleaned)

    return candidates


def extract_preferred_answer(text):
    candidates = extract_answer_candidates(text)
    return candidates[0] if candidates else None


def extract_numeric_answer(text):
    cleaned = clean_answer_text(text)
    if not cleaned:
        return None

    candidates = extract_answer_candidates(cleaned) or [cleaned]
    for candidate in candidates:
        numbers = re.findall(r"-?\d[\d,]*(?:\.\d+)?", candidate)
        if numbers:
            return numbers[-1]
        word_number = number_words_to_numeric_string(candidate)
        if word_number is not None:
            return word_number
    return None

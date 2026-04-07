import ast
import os
import re
from typing import Optional

try:
    from dotenv import load_dotenv as _dotenv_load_dotenv
except Exception:
    _dotenv_load_dotenv = None


_ENV_LINE_RE = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)\s*$")


def _strip_inline_comment(value: str) -> str:
    in_single = False
    in_double = False

    for index, char in enumerate(value):
        if char == "'" and not in_double:
            in_single = not in_single
        elif char == '"' and not in_single:
            in_double = not in_double
        elif char == "#" and not in_single and not in_double:
            return value[:index].rstrip()

    return value.rstrip()


def _parse_env_value(raw_value: str) -> str:
    value = _strip_inline_comment(raw_value.strip())
    if not value:
        return ""

    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        try:
            parsed = ast.literal_eval(value)
            if parsed is None:
                return ""
            return str(parsed)
        except Exception:
            return value[1:-1]

    return value


def _load_dotenv_fallback(dotenv_path: Optional[str], override: bool = False) -> bool:
    if not dotenv_path or not os.path.exists(dotenv_path):
        return False

    loaded_any = False
    with open(dotenv_path, "r", encoding="utf-8") as env_file:
        for line in env_file:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue

            match = _ENV_LINE_RE.match(line)
            if not match:
                continue

            key, raw_value = match.groups()
            if not override and key in os.environ:
                continue

            os.environ[key] = _parse_env_value(raw_value)
            loaded_any = True

    return loaded_any


def load_dotenv(dotenv_path: Optional[str] = None, override: bool = False) -> bool:
    if _dotenv_load_dotenv is not None:
        return bool(_dotenv_load_dotenv(dotenv_path=dotenv_path, override=override))
    return _load_dotenv_fallback(dotenv_path=dotenv_path, override=override)

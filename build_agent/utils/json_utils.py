import json
from typing import Any, Optional


def strip_markdown_fences(text: str) -> str:
    """Strip a single surrounding markdown fence from a model reply."""
    stripped = (text or "").strip()
    if not stripped.startswith("```"):
        return stripped

    first_newline = stripped.find("\n")
    if first_newline == -1:
        return stripped

    body = stripped[first_newline + 1 :]
    if body.endswith("```"):
        body = body[:-3]
    return body.strip()


def _extract_largest_balanced_json(text: str, opener: str, closer: str) -> Optional[str]:
    """Return the largest balanced JSON-looking segment in text."""
    if not text:
        return None

    best: Optional[str] = None
    n = len(text)
    i = 0

    while i < n:
        if text[i] != opener:
            i += 1
            continue

        depth = 0
        in_str = False
        escaped = False
        j = i

        while j < n:
            ch = text[j]
            if in_str:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_str = False
            else:
                if ch == '"':
                    in_str = True
                elif ch == opener:
                    depth += 1
                elif ch == closer:
                    depth -= 1
                    if depth == 0:
                        candidate = text[i : j + 1]
                        if best is None or len(candidate) > len(best):
                            best = candidate
                        break
            j += 1

        i = max(i + 1, j + 1)

    return best


def extract_largest_json_object(text: str) -> Optional[str]:
    return _extract_largest_balanced_json(text, "{", "}")


def extract_largest_json_array(text: str) -> Optional[str]:
    return _extract_largest_balanced_json(text, "[", "]")


def load_json_loose(text: str, expected: str = "any") -> Any:
    """
    Parse JSON from a noisy model reply.

    `expected` can be "any", "object", or "array".
    """
    stripped = strip_markdown_fences(text)
    candidates = [stripped]

    if expected in ("any", "object"):
        obj = extract_largest_json_object(stripped)
        if obj:
            candidates.append(obj)
    if expected in ("any", "array"):
        arr = extract_largest_json_array(stripped)
        if arr:
            candidates.append(arr)

    seen = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        try:
            parsed = json.loads(candidate)
        except Exception:
            continue

        if expected == "object" and not isinstance(parsed, dict):
            continue
        if expected == "array" and not isinstance(parsed, list):
            continue
        return parsed

    raise ValueError(f"Could not parse {expected} JSON payload")

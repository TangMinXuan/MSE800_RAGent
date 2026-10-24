import json
from dataclasses import asdict, is_dataclass
from typing import Any


def to_json(obj: Any, *, indent: int | None = None) -> str:
    """Serialize to JSON, supporting dataclasses and keeping non-ASCII characters."""
    return json.dumps(obj, ensure_ascii=False, indent=indent, default=_default)


def from_json(text: str | None, default: Any = None) -> Any:
    """Parse JSON; return default when the input is empty or invalid."""
    if text is None or not text.strip():
        return default
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return default


def _default(obj: Any) -> Any:
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

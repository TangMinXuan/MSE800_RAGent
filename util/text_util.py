import re


def normalize_whitespace(text: str | None) -> str:
    """Collapse consecutive whitespace into a single space and strip the ends."""
    return re.sub(r"\s+", " ", text or "").strip()


def truncate(text: str | None, max_length: int, suffix: str = "...") -> str:
    text = text or ""
    if max_length <= 0:
        return ""
    if len(text) <= max_length:
        return text
    if max_length <= len(suffix):
        return text[:max_length]
    return text[: max_length - len(suffix)] + suffix


def estimate_tokens(text: str | None) -> int:
    """Roughly estimate the token count (~4 characters per token); for budgeting only."""
    return (len(text or "") + 3) // 4


def split_paragraphs(text: str | None) -> list[str]:
    """Split on blank lines and drop empty paragraphs."""
    return [p.strip() for p in re.split(r"\n\s*\n", text or "") if p.strip()]

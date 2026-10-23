def is_blank(s: str | None) -> bool:
    return s is None or s.strip() == ""

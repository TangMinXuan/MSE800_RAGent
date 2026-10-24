import os


def get_env(name: str, default: str | None = None) -> str | None:
    """Read an environment variable; blank values count as unset."""
    value = (os.getenv(name) or "").strip()
    return value or default


def require_env(name: str) -> str:
    value = get_env(name)
    if value is None:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def get_env_int(name: str, default: int) -> int:
    value = get_env(name)
    return default if value is None else int(value)


def get_env_bool(name: str, default: bool = False) -> bool:
    value = get_env(name)
    return default if value is None else value.lower() in {"1", "true", "yes", "on"}

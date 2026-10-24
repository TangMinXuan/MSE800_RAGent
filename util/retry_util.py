import functools
import time
from typing import Callable, TypeVar

T = TypeVar("T")


def retry(
    attempts: int = 3,
    delay: float = 0.5,
    backoff: float = 2.0,
    exceptions: tuple[type[BaseException], ...] = (Exception,),
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Retry decorator with exponential backoff, e.g. for LLM or Chroma network calls."""
    if attempts < 1:
        raise ValueError("attempts must be >= 1")

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            wait = delay
            for attempt in range(1, attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions:
                    if attempt == attempts:
                        raise
                    time.sleep(wait)
                    wait *= backoff
            raise RuntimeError("unreachable")

        return wrapper

    return decorator

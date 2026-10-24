import time
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Iterator


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_now_iso() -> str:
    return utc_now().isoformat()


@contextmanager
def timer() -> Iterator[callable]:
    """Usage: `with timer() as elapsed: ...`; elapsed() returns the seconds spent so far."""
    start = time.perf_counter()
    yield lambda: time.perf_counter() - start

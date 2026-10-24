from typing import Hashable, Iterable, Iterator, TypeVar

T = TypeVar("T")


def batched(items: Iterable[T], size: int) -> Iterator[list[T]]:
    """Split items into fixed-size batches, e.g. for bulk vector store writes."""
    if size <= 0:
        raise ValueError("size must be positive")
    batch: list[T] = []
    for item in items:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def dedupe(items: Iterable[T], key=None) -> list[T]:
    """Remove duplicates while keeping order; key selects the value to compare."""
    seen: set[Hashable] = set()
    result: list[T] = []
    for item in items:
        k = key(item) if key else item
        if k not in seen:
            seen.add(k)
            result.append(item)
    return result

import hashlib
from pathlib import Path


def read_text(path: str | Path, encoding: str = "utf-8") -> str:
    return Path(path).read_text(encoding=encoding)


def file_sha256(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """Hash the file content, e.g. to detect changed documents and skip re-ingestion."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def list_files(directory: str | Path, suffixes: tuple[str, ...] = (".txt", ".md", ".pdf")) -> list[Path]:
    """Recursively list files with the given suffixes, sorted."""
    suffixes = tuple(s.lower() for s in suffixes)
    return sorted(p for p in Path(directory).rglob("*") if p.is_file() and p.suffix.lower() in suffixes)

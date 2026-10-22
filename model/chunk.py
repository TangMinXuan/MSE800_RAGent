from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    id: int
    content: str
    source_file: str

    def __repr__(self) -> str:
        return f"Chunk{{id={self.id}, content='{self.content}'}}"

# STUB: replaced in the pipeline migration step
from dataclasses import dataclass
from pathlib import Path


@dataclass
class LoadedDocument:
    text: str


class DocumentLoader:
    def load_document(self, file: Path) -> LoadedDocument:
        raise NotImplementedError

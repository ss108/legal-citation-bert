from dataclasses import dataclass
from typing import Dict

from beartype import beartype
from pydantic import BaseModel


class CitationExtractionResult(BaseModel):
    cases: Dict[str, int]
    statutes: Dict[str, int]

    @classmethod
    def from_dict(cls, d: Dict):
        return cls(cases=d["cases"], statutes=d["statutes"])


@beartype
@dataclass(frozen=True)
class TestFile:
    file_text: str
    correct: CitationExtractionResult

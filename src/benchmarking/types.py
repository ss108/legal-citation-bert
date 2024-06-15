from typing import Dict

from pydantic import BaseModel


class CitationExtractionResult(BaseModel):
    cases: Dict[str, int]
    statutes: Dict[str, int]

    @classmethod
    def from_dict(cls, d: Dict):
        return cls(cases=d["cases"], statutes=d["statutes"])

from __future__ import annotations

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

    def sort(self):
        self.cases = dict(sorted(self.cases.items(), key=lambda x: x[0], reverse=True))
        self.statutes = dict(
            sorted(self.statutes.items(), key=lambda x: x[0], reverse=True)
        )
        return self

    def err_count(self, model_response: CitationExtractionResult) -> int:
        err_count = 0

        for k, v in self.cases.items():
            if k not in model_response.cases:
                err_count += v
            else:
                correct_count = v
                response_count = model_response.cases[k]
                err_count += abs(correct_count - response_count)

        for k, v in self.statutes.items():
            if k not in model_response.statutes:
                err_count += v
            else:
                correct_count = v
                response_count = model_response.statutes[k]
                err_count += abs(correct_count - response_count)

        return err_count


@beartype
@dataclass(frozen=True)
class TestFile:
    file_text: str
    correct: CitationExtractionResult

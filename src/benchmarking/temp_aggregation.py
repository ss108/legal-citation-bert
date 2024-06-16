"""
Temporary code to ingest raw model responses.
Goal is to put this logic in a separate library.
Thus, the schema is subject to dramatic change.
"""

from typing import List, Optional, Tuple, TypeAlias

from pydantic import BaseModel

PIN_CITE: TypeAlias = Tuple[int, Optional[int]]


def citation_from(token_label_pairs: List[Tuple[str, str]]): ...


class CaselawCitation(BaseModel):
    case_name: str
    volume: int
    reporter: str

    starting_page: Optional[int]
    pin_cite: Optional[PIN_CITE]

    court: Optional[str]
    year: Optional[int]

    @property
    def is_full(self) -> bool:
        return self.starting_page is not None and (
            "v. " in self.case_name or "v " in self.case_name
        )

    @property
    def plaintiff(self) -> str:
        case_name_components = self.case_name.split(" v. ")
        return case_name_components[0]

    @property
    def defendant(self) -> str:
        case_name_components = self.case_name.split(" v. ")
        return case_name_components[1]

    @property
    def full_text(self) -> str:
        components = [self.case_name, ","]


class StatuteCitation(BaseModel):
    title: Optional[str]
    code: str
    section: Optional[str]

    year: Optional[int]

    @property
    def full_text(self) -> str:
        components = []

        if self.title:
            components.append(self.title)

        components.append(self.code)

        if self.section:
            components.append("§")
            components.append(self.section)

        return " ".join(components)

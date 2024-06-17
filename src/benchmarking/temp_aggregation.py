"""
Temporary code to ingest raw model responses.
Goal is to put this logic in a separate library.
Thus, the schema is subject to dramatic change.
"""

from typing import List, NamedTuple, Optional, Protocol, Tuple, Type, TypeAlias

from pydantic import BaseModel

PIN_CITE: TypeAlias = Tuple[int, Optional[int]]


class LabelPrediction(NamedTuple):
    token: str
    label: str


class ICitation(Protocol):
    @classmethod
    def from_token_label_pairs(cls, token_label_pairs: List[LabelPrediction]): ...

    @property
    def full_text(self) -> str: ...


def citation_from(token_label_pairs: List[LabelPrediction]) -> Optional[ICitation]:
    labels_only = [pair.label for pair in token_label_pairs]

    citation_class: Optional[Type[ICitation]] = None

    match labels_only:
        case ["B-TITLE", *rest]:
            ...
        case ["B-CASE_NAME", _]:
            ...


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
        components = [f"{self.case_name},", self.volume, self.reporter]

        if self.starting_page:
            components.append(self.starting_page)

        if self.pin_cite:
            s, e = self.pin_cite
            components.append(f", {s}")

            if e:
                components.append(
                    f"-{e}"
                )  # TODO: ensure this is the right type of dash

        paren_block = ""
        if self.court and self.year:
            paren_block = f"({self.court} {self.year})"
        elif self.court and self.year is None:
            paren_block = f"({self.court})"
        elif self.court is None and self.year:
            paren_block = f"({self.year})"

        components.append(paren_block)

        return " ".join(components)


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

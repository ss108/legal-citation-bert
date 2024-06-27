"""
Temporary code to ingest raw model responses.
Goal is to put this logic in a separate library.
Thus, the schema is subject to dramatic change.
"""

from __future__ import annotations

from typing import List, NamedTuple, Optional, Protocol, Tuple, Type, TypeAlias

from pydantic import BaseModel

PIN_CITE: TypeAlias = Tuple[int, Optional[int]]


class LabelPrediction(NamedTuple):
    token: str
    label: str

    def __repr__(self):
        return f"{self.token}: {self.label}"


class ICitation(Protocol):
    @classmethod
    def from_token_label_pairs(
        cls, token_label_pairs: List[LabelPrediction]
    ) -> Optional[ICitation]: ...

    @property
    def full_text(self) -> str: ...


def citation_from(token_label_pairs: List[LabelPrediction]) -> Optional[ICitation]:
    labels_only = [pair.label for pair in token_label_pairs]

    citation_class: Optional[Type[ICitation]] = None

    match labels_only:
        case ["B-TITLE", *rest]:
            ...
            # citation_class = StatuteCitation
        case ["B-CASE_NAME", *rest]:
            citation_class = CaselawCitation
        case _:
            return None


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

    @classmethod
    def from_token_label_pairs(
        cls, token_label_pairs: List[LabelPrediction]
    ) -> Optional[CaselawCitation]:
        case_name = ""
        volume = None
        reporter = ""
        starting_page = None
        pin_cite = None
        court = None
        year = None

        for pair in token_label_pairs:
            token, label = pair

            if label == "B-CASE_NAME":
                case_name += token + " "
            elif label == "B-VOLUME":
                volume = int(token)
            elif label == "B-REPORTER":
                reporter = token
            elif label == "B-PAGE":
                starting_page = int(token)
            elif label == "B-PIN":
                pin_cite = (int(token), None)
            elif label == "I-PIN":
                pin_cite = (pin_cite[0], int(token))
            elif label == "B-COURT":
                court = token
            elif label == "B-YEAR":
                year = int(token)

        if volume is None:
            return None

        return cls(
            case_name=case_name.strip(),
            volume=volume,
            reporter=reporter,
            starting_page=starting_page,
            pin_cite=pin_cite,
            court=court,
            year=year,
        )


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


def aggregate_entities(labels: List[LabelPrediction]) -> List[LabelPrediction]:
    """
    Aggregates entities that are split into multiple tokens.
    """
    aggregated: List[LabelPrediction] = []

    current_entity = ""
    current_label = ""

    for token, label in labels:
        if label.startswith("B-"):
            if current_entity:
                aggregated.append(
                    LabelPrediction(token=current_entity, label=current_label)
                )

            current_entity = token
            current_label = label
        elif label.startswith("I-"):
            current_entity += token
        elif label.startswith("O"):
            if current_entity:
                aggregated.append(
                    LabelPrediction(token=current_entity, label=current_label)
                )

            current_entity = ""
            current_label = ""

    if current_entity:
        aggregated.append(LabelPrediction(token=current_entity, label=current_label))

    return aggregated

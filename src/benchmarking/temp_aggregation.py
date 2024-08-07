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

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, LabelPrediction):
            return False

        return self.token == value.token and self.label == value.label


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
        case ["B-CASE_NAME", *rest]:  # noqa
            citation_class = CaselawCitation
        case _:
            return None

    if not citation_class:
        return None

    citation_class.from_token_label_pairs(token_label_pairs)


class CaselawCitation(BaseModel):
    case_name: str
    volume: int
    reporter: str

    starting_page: Optional[int]
    raw_pin_cite: Optional[str] = None

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
    def pin_cite(self) -> Optional[PIN_CITE]:
        if self.raw_pin_cite is None:
            return None

        if "-" in self.raw_pin_cite:
            start, end = self.raw_pin_cite.split("-")
            return int(start), int(end)

        return int(self.raw_pin_cite), None

    @property
    def guid(self) -> str:
        base = f"{self.volume} {self.reporter}"
        if self.starting_page:
            base += f" {self.starting_page}"

        return base

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
        raw_pin_cite = None
        court = None
        year = None

        for token, label in token_label_pairs:
            if label == "CASE_NAME":
                case_name += token
            elif label == "VOLUME":
                volume = int(token)
            elif label == "REPORTER":
                reporter = token
            elif label == "PAGE":
                starting_page = int(token)
            elif label == "PIN":
                raw_pin_cite = token
            # elif label == "I-PIN":
            #     assert raw_pin_cite is not None, "Encountered I-PIN without B-PIN"
            #     raw_pin_cite += _process_token(token)
            elif label == "COURT":
                court = token
            elif label == "YEAR":
                year = int(token)

        if volume is None:
            return None

        return cls(
            case_name=case_name.strip(),
            volume=volume,
            reporter=reporter,
            starting_page=starting_page,
            raw_pin_cite=raw_pin_cite,
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


def _process_label(label: str) -> str:
    if label.startswith("B-"):
        return label[2:]
    elif label.startswith("I-"):
        return label[2:]
    else:
        raise ValueError(f"Unexpected label: {label}")


def _process_token(token: str) -> str:
    if token.startswith("##"):
        return token[2:]

    return token


def aggregate_entities(labels: List[LabelPrediction]) -> List[LabelPrediction]:
    """
    Take the raw model output and "squash" the tokens into entities.
    E.g. a series of tokens with labels like: B-CASE_NAME, I-CASE_NAME,
    I_CASE_NAME should all be aggregated into one LabelPrediction with label CASE_NAME
    """
    aggregated: List[LabelPrediction] = []

    current_entity = ""
    current_label = ""

    for token, label in labels:
        if token[0].isupper() and len(current_entity) > 0 and "CASE_NAME" in label:
            token = f" {token}"

        if label.startswith("B-"):
            if current_entity:
                aggregated.append(
                    LabelPrediction(token=current_entity, label=current_label)
                )

            current_entity = token
            current_label = _process_label(label)
        elif label.startswith("I-"):
            current_entity += _process_token(token)
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

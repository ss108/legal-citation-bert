from enum import Enum
from typing import List, Literal, Tuple, TypeAlias, TypedDict

from pydantic import BaseModel

STATEMENT_TYPE: TypeAlias = Literal["fact", "law"]


class Sentence(TypedDict):
    text: str


class Datum(BaseModel):
    text: str
    tokens: List[str]
    tags: List[Tuple[str, str]]

    @classmethod
    def empty(cls):
        return cls(text="", tokens=[], tags=[])


class CIT_TYPE(Enum):
    CASE = 0
    STATUTE = 1


class CIT_FORM(Enum):
    SHORT = 0
    LONG = 1


class DataGenerationArgs(BaseModel):
    cit_type: CIT_TYPE
    cit_form: CIT_FORM
    number: int

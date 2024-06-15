from typing import Dict, Set

from pydantic import BaseModel


class CitationExtractionFormat(BaseModel):
    cases: Dict[str, str]
    statutes: Set[str]

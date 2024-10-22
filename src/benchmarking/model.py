from typing import Dict

from cit_parser import Authorities

from .types import CitationExtractionResult


def authorities_to_citation_extraction_result(
    authorities: Authorities,
) -> CitationExtractionResult:
    cases: Dict[str, int] = {}
    statutes: Dict[str, int] = {}

    for full_citation, citations_list in authorities.caselaw.items():
        full_text = full_citation.full_text
        cases[full_text] = len(citations_list)

    for full_citation, citations_list in authorities.statutes.items():
        full_text = full_citation.full_text
        statutes[full_text] = len(citations_list)

    return CitationExtractionResult(cases=cases, statutes=statutes)

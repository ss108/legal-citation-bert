from .llm import extract_citations
from .types import CitationExtractionFormat


async def score_llm(
    text: str,
    correct: CitationExtractionFormat,
    llm_extraction: CitationExtractionFormat,
) -> int:
    error_count = 0

    for k in correct.cases.keys():
        if not llm_extraction.cases.get(k):
            error_count += 1

    for s in correct.statutes:
        if s not in llm_extraction.statutes:
            error_count += 1

    return error_count

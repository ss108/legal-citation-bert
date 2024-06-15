from .llm import IterativeCitExtraction
from .types import CitationExtractionResult


def llm_err_count(
    *,
    correct: CitationExtractionResult,
    llm_extraction: IterativeCitExtraction,
) -> int:
    error_count = 0

    for k in correct.cases.keys():
        if not llm_extraction.cases.get(k):
            error_count += 1

    for s in correct.statutes:
        if s not in llm_extraction.statutes:
            error_count += 1

    return error_count

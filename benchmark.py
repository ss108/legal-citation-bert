import asyncio

from wasabi import msg

from src.benchmarking.items import TEST_ITEMS
from src.benchmarking.llm import llm_extract_citations_from_document


def run_llm_extraction():
    for text, c in TEST_ITEMS:
        res = asyncio.run(llm_extract_citations_from_document(text))
        msg.info(res.dict())

        err_count = c.err_count(res)
        if err_count > 3:
            msg.fail(f"LLM error count: {c.err_count(res)}")
        else:
            msg.good(f"LLM error count: {c.err_count(res)}")


run_llm_extraction()
"""
from src.benchmarking.types import CitationExtractionResult

correct = CitationExtractionResult.from_dict(
    {
        "cases": {
            "588 F.3d 97": 1,
            "604 F.2d 200": 2,
            "335 F.3d 141": 1,
            "954 F. Supp. 2d 145": 2,
        },
        "statutes": {
            "15 U.S.C. § 1114": 1,
        },
    }
)


"""

import asyncio

from cit_parser import Authorities, invoke, organize
from wasabi import msg

from src.benchmarking.items import TEST_ITEMS
from src.benchmarking.llm import llm_extract_citations_from_document
from src.benchmarking.model import authorities_to_citation_extraction_result
from src.benchmarking.types import BenchmarkResult, CitationExtractionResult


async def run_llm_extraction() -> BenchmarkResult:
    tasks = [
        llm_extract_citations_for_item(text, correct_citation)
        for text, correct_citation in TEST_ITEMS
    ]

    results = await asyncio.gather(*tasks)

    combined_result = BenchmarkResult.combine(results)

    return combined_result


async def llm_extract_citations_for_item(
    text: str, correct: CitationExtractionResult
) -> BenchmarkResult:
    res = await llm_extract_citations_from_document(text)

    err_count = correct.err_count(res)
    correct_count = (
        sum(correct.cases.values()) + sum(correct.statutes.values()) - err_count
    )
    benchmark_result = BenchmarkResult("LLM")
    benchmark_result.add_result(text, correct_count, err_count)

    return benchmark_result


def run_model_extraction() -> BenchmarkResult:
    benchmark_result = BenchmarkResult("BERT Model")

    for text, correct_citation in TEST_ITEMS:
        res = invoke(text)
        auth: Authorities = organize(res)

        formatted_result: CitationExtractionResult = (
            authorities_to_citation_extraction_result(auth)
        )
        msg.info(formatted_result.dict())

        err_count = correct_citation.err_count(formatted_result)
        correct_count = (
            sum(correct_citation.cases.values())
            + sum(correct_citation.statutes.values())
            - err_count
        )
        benchmark_result.add_result(text, correct_count, err_count)

    return benchmark_result


if __name__ == "__main__":
    msg.info("Running LLM extraction...")
    llm_res = asyncio.run(run_llm_extraction())

    msg.info("Running model extraction...")
    model_res = run_model_extraction()

    # model_res.log_individual_results()
    model_res.log_overall_results()

    # llm_res.log_individual_results()
    llm_res.log_overall_results()

    # relevant_item, c = TEST_ITEMS[-1]
    # res = asyncio.run(llm_extract_citations_for_item(relevant_item, c))
    # print(res)

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

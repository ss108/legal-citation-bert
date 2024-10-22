import asyncio

from wasabi import msg

from src.benchmarking.llm import llm_extract_citations_from_document


def run_llm_extraction():
    for file_name, data in td.items():
        res = asyncio.run(llm_extract_citations_from_document(data.file_text))
        msg.info("final res")
        print(res.sort())

        msg.warn(f"correct: {data.correct.sort()}")
        msg.fail(f"LLM error count for {file_name}: {data.correct.err_count(res)}")


run_llm_extraction()

# chunks = split_text("look at this pin cite: Foo v. Bar, 551 U. S. 877, 904-907 (2007).")
# model = get_model()

# for c in chunks:
#     print(c)
#     predictions = get_labels(c, model)
#     print(predictions)

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

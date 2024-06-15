import asyncio

from src.benchmarking.llm import extract_citations

test_text = """This brief is concerned with the language and
interpretation of what we call the rescission
condition in 8 U.S.C. § 1229a(b)(5)(C)(ii), under
which an order of removal entered in absentia “may be
rescinded.”"""

res = asyncio.run(extract_citations(test_text))
print(res)

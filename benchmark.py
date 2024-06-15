import asyncio

from src.benchmarking.llm import extract_citations_from_chunks

test_text1 = """This brief is concerned with the language and
interpretation of what we call the rescission
condition in 8 U.S.C. § 1229a(b)(5)(C)(ii), under
which an order of removal entered in absentia “may be
rescinded.”"""

test_text2 = """
YEHEYHEYEH. 8 U.S.C. § 1229a(b)(5)(C)(ii) son born and raised
"""

res = asyncio.run(extract_citations_from_chunks([test_text1, test_text2]))

print(res)
assert res.statutes["8 U.S.C. § 1229a(b)(5)(C)(ii)"] == 2

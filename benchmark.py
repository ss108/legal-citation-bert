import asyncio
import importlib.util
import sys
from pathlib import Path

from wasabi import msg

from src.benchmarking.llm import extract_citations_from_document
from src.benchmarking.test_files.loader import get_test_data

TEST_FILES_DIR = Path(__file__).parent / "src" / "benchmarking" / "test_files"


def load_module_from_file(file_path: str):
    module_name = Path(file_path).stem
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec and spec.loader:
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    return None


td = get_test_data()


for file_name, data in td.items():
    res = asyncio.run(extract_citations_from_document(data.file_text))
    msg.info("final res")
    print(res.sort())

    msg.warn(f"correct: {data.correct.sort()}")
    msg.fail(f"Diff: {data.correct.err_count(res)}")
# llm_errs = llm_err_count(correct=data.correct, llm_extraction=res)
# print(f"File: {file_name}, LLM Error Count: {llm_errs}")

# chunks = split_text("look at this pin cite: Foo v. Bar, 551 U. S. 877, 904-907 (2007).")
# model = get_model()

# for c in chunks:
#     print(c)
#     predictions = get_labels(c, model)
#     print(predictions)

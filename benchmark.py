import glob
import importlib.util
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import fitz
from beartype import beartype

from src.benchmarking.model import get_labels, split_text
from src.benchmarking.types import CitationExtractionResult

TEST_FILES_DIR = Path(__file__).parent / "src" / "benchmarking" / "test_files"


@beartype
@dataclass(frozen=True)
class TestFile:
    file_text: str
    correct: CitationExtractionResult


def load_module_from_file(file_path: str):
    module_name = Path(file_path).stem
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec and spec.loader:
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    return None


def get_test_data():
    test_files_dir = Path(TEST_FILES_DIR).resolve()

    test_data: Dict[str, TestFile] = {}

    for subdir, dirs, files in os.walk(test_files_dir):
        print(f"Processing directory: {subdir}")

        # Check for correct.py and text.pdf in the current directory
        correct_files = glob.glob(os.path.join(subdir, "correct.py"))
        text_files = glob.glob(os.path.join(subdir, "text.pdf"))

        print("Correct files:", correct_files)
        print("Text files:", text_files)

        if correct_files and text_files:
            assert (
                len(correct_files) == 1
            ), "There must be exactly one correct.py file per test case"
            assert (
                len(text_files) == 1
            ), "There must be exactly one text.pdf file per test case"

            correct_file = correct_files[0]
            text_file = text_files[0]

            # Example for loading correct.py (Assume it defines a variable 'correct')
            module = load_module_from_file(correct_file)
            correct = getattr(module, "correct", None)
            assert isinstance(
                correct, CitationExtractionResult
            ), "correct.py must define a variable 'correct' of type CitationExtractionResult"

            page_text = []
            with fitz.open(text_file) as doc:
                for page in doc:
                    page_text.append(page.get_text())  # pyright: ignore

            folder_name = Path(subdir).name
            test_data[folder_name] = TestFile(
                file_text="\n".join(page_text), correct=correct
            )

        else:
            print(f"Skipping directory {subdir} as it lacks the required files.")

    return test_data


# td = get_test_data()

# for file_name, data in td.items():
#     res = asyncio.run(extract_citations_from_document(data.file_text))
#     print(res)
#     llm_errs = llm_err_count(correct=data.correct, llm_extraction=res)
#     print(f"File: {file_name}, LLM Error Count: {llm_errs}")

chunks = split_text("This is a test sentence. This is like a flashback, a drim.")
print(chunks[0])
predictions = get_labels(chunks[0])

print(predictions)

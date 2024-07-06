import os
from importlib import import_module
from pathlib import Path
from typing import Dict

import fitz

from src.benchmarking.types import TestFile

TEST_FILES_DIR = Path(__file__).parent

IGNORE_DIRS = {"__pycache__"}


def _process_pdf(pdf_path: str) -> str:
    page_text = []
    with fitz.open(pdf_path) as doc:
        for page in doc:
            page_text.append(page.get_text())  # pyright: ignore

    return "\n".join(page_text)


def get_test_data() -> Dict[str, TestFile]:
    test_data = {}

    test_files_dir = TEST_FILES_DIR.resolve()

    for subdir in os.scandir(test_files_dir):
        if subdir.is_dir():
            name = subdir.name
            if name not in IGNORE_DIRS:
                pdf_text = None
                correct = None
                for file in os.scandir(subdir):
                    if file.is_file():
                        if file.name == "text.pdf":
                            pdf_text = _process_pdf(file.path)
                        elif file.name == "correct.py":
                            as_module = import_module(
                                f"src.benchmarking.test_files.{name}.correct"
                            )
                            correct = getattr(as_module, "correct")

                if pdf_text and correct:
                    test_data[name] = TestFile(file_text=pdf_text, correct=correct)

    return test_data

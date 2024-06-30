import glob
import importlib.util
import os
from importlib import import_module
from pathlib import Path

TEST_FILES_DIR = Path(__file__).parent

IGNORE_DIRS = {"__pycache__"}


def enumerate_subdirs():
    test_files_dir = TEST_FILES_DIR.resolve()

    for subdir in os.scandir(test_files_dir):
        if subdir.is_dir():
            name = subdir.name
            if name not in IGNORE_DIRS:
                print(subdir.name)

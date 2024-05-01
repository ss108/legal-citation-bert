from pathlib import Path
from typing import List

import pandas as pd
from datasets import DatasetDict

from .types import Datum

DATASET_DIR = Path(__file__).parent / "datasets"


def get_next_version() -> str:
    version_dirs = [
        d for d in DATASET_DIR.iterdir() if d.is_dir() and d.name.startswith("v")
    ]

    version_numbers = sorted(
        [int(d.name[1:]) for d in version_dirs if d.name[1:].isdigit()]
    )

    next_version_number = version_numbers[-1] + 1 if version_numbers else 1

    next_version_dir = f"v{next_version_number}"

    return next_version_dir


def store_raw_dataset(data: List[Datum], overwrite_version: str | None):
    """Store unsegmented data."""
    dir_name = get_next_version() if not overwrite_version else overwrite_version
    df = pd.DataFrame(
        [d.dict() for d in data]
    )  # Ensure Datum objects have a .dict() method or adjust accordingly

    target_dir = DATASET_DIR / dir_name
    target_dir.mkdir(
        parents=True, exist_ok=True
    )  # Create the directory if it does not exist

    file_path = target_dir / f"full_data_{dir_name}.json"
    df.to_json(file_path, orient="records", lines=True)


def _make_formatted_dataset_path(version: str) -> str:
    target_dir = DATASET_DIR / version
    return f"{target_dir}/formatted_data_{version}"


def store_formatted_dataset(ds: DatasetDict, version: str):
    ds.save_to_disk(_make_formatted_dataset_path(version))


def get_full_dataset(version: str | None) -> pd.DataFrame:
    if version is None:
        version = get_next_version()

    file_path = DATASET_DIR / version / f"full_data_{version}.json"
    df = pd.read_json(file_path)
    return df


def get_formatted_dataset(version: str) -> DatasetDict:
    ds = DatasetDict.load_from_disk(_make_formatted_dataset_path(version))
    return ds

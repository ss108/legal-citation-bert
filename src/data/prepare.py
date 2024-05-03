import asyncio
import shutil
import glob
import time
from pathlib import Path
from typing import Dict, List

import pandas as pd
import spacy
from datasets import Dataset, DatasetDict, load_dataset
from wasabi import msg

from src.training.model import LABEL_MAP, get_tokenizer

from .generate import generate, generate_tags
from .manage_datasets import (
    get_formatted_dataset,
)
from .types import DataGenerationArgs, Datum, Sentence

RAW_DATA_DIR = Path(__file__).parent.parent.parent / "raw_data"
PREPARED_DATA_DIR = Path(__file__).parent.parent.parent / "prepared_data"

HF_CACHE_DIR = "/hf_cache"


async def do_sentences(a: DataGenerationArgs):
    res: List[Sentence] = await generate(a)

    data: List[Datum] = await sents_to_data(res)

    df = pd.DataFrame([x.dict() for x in data])
    file_name = f"{RAW_DATA_DIR}/sentences_gen_{a.cit_type.name}_{a.cit_form.name}_{int(time.time())}.jsonl"

    msg.info(f"Saving to {file_name}")

    df.to_json(
        file_name,
        orient="records",
        lines=True,
    )


def retrieve_formatted_dataset(version: str) -> DatasetDict:
    return get_formatted_dataset(version)


CACHED_RAW_DATA_DIR = Path(HF_CACHE_DIR) / "raw_data"


def save_cl_docket_entries_ds():
    dataset = load_dataset(
        "pile-of-law/pile-of-law", "courtlistener_docket_entry_documents"
    )

    msg.info("About to split")

    train_indices = dataset["train"].train_test_split(  # pyright: ignore
        test_size=0.015,
    )

    train_indices["test"].save_to_disk(CACHED_RAW_DATA_DIR / "cl_docket_entries")


def load_raw_cl_docket_entries_ds() -> Dataset:
    return Dataset.load_from_disk(CACHED_RAW_DATA_DIR / "cl_docket_entries")  # pyright: ignore


async def do_tags_for_sentence(text: str, tokens: List[str]) -> Datum:
    tags_object = await generate_tags(text, tokens)

    if not tags_object:
        return Datum.empty()

    tags_map = tags_object["tags"]

    if not isinstance(tags_map, list):
        return Datum.empty()
    elif len(tags_map) != len(tokens):
        return Datum.empty()

    for item in tags_map:
        if not isinstance(item, list):
            return Datum.empty()

        if len(item) == 1:
            return Datum.empty()
    try:
        return Datum(text=text, tokens=tokens, tags=tags_map)
    except Exception as e:
        print("A sadge happened!\n")
        print(tags_map)
        return Datum.empty()


async def sents_to_data(sentences: List[Sentence]) -> List[Datum]:
    tokenizer = get_tokenizer()

    tag_tasks = []
    for sent in sentences:
        text = sent["text"]
        tokens = tokenizer.tokenize(text)
        tag_tasks.append(do_tags_for_sentence(text, tokens))

    res = await asyncio.gather(*tag_tasks)
    filtered_res = [r for r in res if len(r.text) > 0]
    return filtered_res


def _concat_sents(sents: List[str]) -> str:
    return " ".join(sents)


async def text_to_data(text: str) -> List[Datum]:
    tokenizer = get_tokenizer()

    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)

    sentences: List[str] = [sent.text for sent in doc.sents]

    tasks = []

    for i in range(0, len(sentences), 3):
        batch = sentences[i : i + 3]
        text = _concat_sents(batch)
        tokens = tokenizer.tokenize(text)
        tasks.append(do_tags_for_sentence(text, tokens))

    return await asyncio.gather(*tasks)


async def gather_wrapper(tasks):
    await asyncio.gather(*tasks)


async def process_cl_doc(doc: Dict):
    try:
        text = doc["text"]
        url = doc["url"]
        doc_id = url.split("/")[-2]
        msg.info(f"Processing doc {doc_id}")

        data = await text_to_data(text)

        df = pd.DataFrame([x.dict() for x in data])
        df.to_json(
            f"{RAW_DATA_DIR}/cl_{doc_id}.jsonl",
            orient="records",
            lines=True,
        )
    except Exception as e:
        msg.fail(f"Error processing doc {doc_id}:\n{e}")
        raise e


def get_jsonl_files():
    json_files = glob.glob(str(RAW_DATA_DIR / "*.jsonl"))
    return json_files


def create_candidate_dataset(version="v0"):
    jsonl_files = get_jsonl_files()
    combined_df = pd.DataFrame()  # Start with an empty dataframe
    dfs = []

    for file in jsonl_files:
        df = pd.read_json(file, lines=True)
        filtered_df = df[
            df["text"].str.strip().ne("")
            & df["tokens"].map(bool)
            & df["tags"].map(bool)
        ]

        def validate_labels(row):
            for _, label in row["tags"]:
                if label not in LABEL_MAP:
                    print(
                        f"Invalid label found: {label} in text: {row['text']}; file: {file}"
                    )
                    return False
            return True

        # Apply label validation
        filtered_df = filtered_df[filtered_df.apply(validate_labels, axis=1)]
        dfs.append(filtered_df)

    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df.to_json(
        f"{PREPARED_DATA_DIR}/{version}/candidate.jsonl",
        orient="records",
        lines=True,
    )


def load_candidate_ds(version: str = "v0"):
    path = PREPARED_DATA_DIR / version / "candidate.jsonl"
    df = pd.read_json(path, lines=True)
    return Dataset.from_pandas(df)


def split_and_save(ds: Dataset, version: str = "v0"):
    splits = ds.train_test_split(test_size=0.20)

    train = splits["train"]
    train_valid_split = train.train_test_split(test_size=0.1)

    train_dataset = train_valid_split["train"]
    valid_dataset = train_valid_split["test"]
    test_dataset = splits["test"]

    final_splits = DatasetDict(
        {"train": train_dataset, "valid": valid_dataset, "test": test_dataset}
    )
    print(final_splits)

    output_dir = Path(HF_CACHE_DIR) / version
    output_dir.mkdir(parents=True, exist_ok=True)

    final_splits.save_to_disk(output_dir / "ds")


def load_for_training(version: str = "v0") -> DatasetDict:
    path = Path(HF_CACHE_DIR) / version / "ds"
    ds_dict = DatasetDict.load_from_disk(path)
    tokenizer = get_tokenizer()

    def tokenize_and_align_labels(row):
        # Tokenize the input text
        encoding = tokenizer.encode_plus(
            row["text"],
            add_special_tokens=True,
            truncation=True,
            padding="max_length",
            max_length=512,
            return_attention_mask=True,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].squeeze()  # pyright: ignore
        attention_mask = encoding["attention_mask"].squeeze()  # pyright: ignore

        # Map labels and apply the same padding/truncation
        labels = [-100] + [LABEL_MAP[label] for _, label in row["tags"]] + [-100]
        padding_length = len(input_ids) - len(labels)
        labels += [-100] * padding_length

        token_label_pairs = [
            (tokenizer.decode([id]), label) for id, label in zip(input_ids, labels)
        ]

        for token, label in token_label_pairs:
            if token in ["[CLS]", "[SEP]", "[PAD]"]:
                assert label == -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    ds_dict = ds_dict.map(tokenize_and_align_labels, batched=False)
    return ds_dict


def load_final_splits():
    return DatasetDict.load_from_disk(Path(HF_CACHE_DIR) / "final_splits_v0")  # pyright: ignore


def delete_from_cache(name: str):
    shutil.rmtree(Path(HF_CACHE_DIR) / name)

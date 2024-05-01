import asyncio
import json
import time
from typing import Dict, List, Tuple

import pandas as pd
import torch
import typer
from datasets import Dataset, DatasetDict
from eyecite import get_citations
from wasabi import msg

from src.data.inspect import inspect_cl
from src.data.prepare import (
    create_candidate_dataset,
    delete_from_cache,
    do_sentences,
    gather_wrapper,
    get_formatted_dataset,
    load_candidate_ds,
    load_for_training,
    load_raw_cl_docket_entries_ds,
    process_cl_doc,
    save_cl_docket_entries_ds,
    split_and_save,
)
from src.data.types import CIT_FORM, CIT_TYPE, DataGenerationArgs, Datum, Sentence
from src.openai import get_logit_bias
from src.training.model import (
    ALL_LABELS,
    get_base_model,
    get_tokenizer,
    load_model_from_checkpoint,
)
from src.training.train import test_predict, train_model

app = typer.Typer()


@app.command()
def save_ds():
    save_cl_docket_entries_ds()


@app.command()
def gen_sentences():
    args = DataGenerationArgs(cit_form=CIT_FORM.LONG, cit_type=CIT_TYPE.CASE, number=50)
    asyncio.run(do_sentences(args))  # pyright: ignore


@app.command()
def delete_cached_file(name: str):
    delete_from_cache(name)


@app.command()
def store_model():
    get_base_model()


@app.command()
def prepare_candidate_dataset():
    create_candidate_dataset()


@app.command()
def save_hf_ds(version: str = "v0"):
    ds = load_candidate_ds(version)
    split_and_save(ds, version)


@app.command()
def train():
    ds = load_for_training()
    _, trainer = train_model(ds)
    test_predict(trainer, ds["test"])


@app.command()
def download_cl():
    save_cl_docket_entries_ds()


@app.command()
def process_cl_docs():
    docs: Dataset = load_raw_cl_docket_entries_ds()
    print(docs.column_names)
    subset = docs.select(range(33, 36))

    tasks = []
    for d in subset:
        tasks.append(process_cl_doc(d))  # pyright: ignore

    asyncio.run(gather_wrapper(tasks))


@app.command()
def test_model():
    device = torch.device("cuda")
    model = load_model_from_checkpoint()
    model = model.to(device)
    # print(model)

    tokenizer = get_tokenizer()
    text = (
        "which? Scheuer v. Rhodes, 416 U.S. 232, 94 S.Ct. 1683, 40 L.Ed.2d 90 (1974)."
    )

    tokenized_input = tokenizer(text, return_tensors="pt", padding=True)

    # print("Tokenized Input:", tokenized_input)
    # Convert token IDs back to tokens for easier readability
    # tokens = tokenizer.convert_ids_to_tokens(tokenized_input["input_ids"][0])
    # print("Tokens:", tokens)

    tokenized_input = {k: v.to(device) for k, v in tokenized_input.items()}

    model.eval()

    with torch.no_grad():
        outputs = model(**tokenized_input)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)

    # Print the predicted labels
    predicted_labels = [ALL_LABELS[p] for p in predictions[0].tolist()]
    tokens = tokenizer.convert_ids_to_tokens(tokenized_input["input_ids"][0])

    for token, label in zip(tokens, predicted_labels):
        print(f"{token} - {label}")


@app.command()
def hi():
    print("hi")


if __name__ == "__main__":
    app()

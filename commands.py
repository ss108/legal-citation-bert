import asyncio

import torch
import typer
from datasets import Dataset
from transformers import BertTokenizerFast
from wasabi import msg

from src.benchmarking.model import get_labels, get_model, split_text
from src.benchmarking.temp_aggregation import (
    CaselawCitation,
    LabelPrediction,
    aggregate_entities,
    citations_from,
)
from src.data.generate import generate_tags, generate_unofficial_citation
from src.data.prepare import (
    create_candidate_dataset,
    delete_from_cache,
    do_sentences,
    gather_wrapper,
    load_candidate_ds,
    load_for_training,
    load_raw_cl_docket_entries_ds,
    process_cl_doc,
    save_cl_docket_entries_ds,
    split_and_save,
)
from src.data.types import CIT_FORM, CIT_TYPE, DataGenerationArgs
from src.training.model import (
    ALL_LABELS,
    MODEL_NAME,
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
    args = DataGenerationArgs(
        cit_form=CIT_FORM.SHORT, cit_type=CIT_TYPE.CASE, number=20
    )
    asyncio.run(do_sentences(args))  # pyright: ignore


@app.command()
def test_gen():
    res = asyncio.run(generate_unofficial_citation(2))
    print(res)


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
def save_training_ds(version: str = "v0"):
    create_candidate_dataset(version)
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
    subset = docs.select(range(57, 59))

    tasks = []
    for d in subset:
        tasks.append(process_cl_doc(d))  # pyright: ignore

    asyncio.run(gather_wrapper(tasks))


@app.command()
def test_mistral_labeling():
    text = "The quick brown fox jumps over the lazy dog. Hox v. Doe, 123 U.S. 456, 499 (2021)."
    tokenizer = get_tokenizer()

    tokens = tokenizer.tokenize(text)
    res = asyncio.run(generate_tags(text, tokens))
    print(res)


@app.command()
def test_from():
    labels = [
        LabelPrediction(token="[CLS]", label="O"),
        LabelPrediction(token="Moreover", label="O"),
        LabelPrediction(token=",", label="O"),
        LabelPrediction(token="the", label="O"),
        LabelPrediction(token="California", label="O"),
        LabelPrediction(token="Civil", label="B-CODE"),
        LabelPrediction(token="Code", label="I-CODE"),
        LabelPrediction(token="§", label="O"),
        LabelPrediction(token="210", label="B-SECTION"),
        LabelPrediction(token="##0", label="I-SECTION"),
        LabelPrediction(token="explicitly", label="O"),
        LabelPrediction(token="states", label="O"),
        LabelPrediction(token="that", label="O"),
        LabelPrediction(token='"', label="O"),
        LabelPrediction(token="all", label="O"),
        LabelPrediction(token="property", label="O"),
        LabelPrediction(token="owners", label="O"),
        LabelPrediction(token="owe", label="O"),
        LabelPrediction(token="en", label="O"),
        LabelPrediction(token="##tra", label="O"),
        LabelPrediction(token="##nts", label="O"),
        LabelPrediction(token="a", label="O"),
        LabelPrediction(token="basic", label="O"),
        LabelPrediction(token="level", label="O"),
        LabelPrediction(token="of", label="O"),
        LabelPrediction(token="care", label="O"),
        LabelPrediction(token=".", label="O"),
        LabelPrediction(token="[SEP]", label="O"),
    ]

    res = citations_from(labels)
    print(res)


@app.command()
def test_model():
    model = get_model()

    text = """City U. L.R. 551, 589 (2015) (quoting Schane v.  Int’l Bhd. of
Teamsters Union Local No. 710 Pension  Fund Pension Plan, 760 F.3d 585, 589–90
(7th Cir.  2014))."""

    sentences = split_text(text)

    for s in sentences:
        res = get_labels(s, model)
        print(res)


@app.command()
def push_to_hub():
    model = load_model_from_checkpoint()
    model.push_to_hub("ss108/legal-citation-bert", use_temp_dir=True)  # pyright: ignore

    tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)
    tokenizer.push_to_hub("ss108/legal-citation-bert", use_temp_dir=True)  # pyright: ignore


@app.command()
def hi():
    print("hi")


if __name__ == "__main__":
    app()

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
        LabelPrediction(token='"', label="O"),
        LabelPrediction(token="Con", label="O"),
        LabelPrediction(token="##fusion", label="O"),
        LabelPrediction(token='"', label="O"),
        LabelPrediction(token="in", label="O"),
        LabelPrediction(token="this", label="O"),
        LabelPrediction(token="context", label="O"),
        LabelPrediction(token="is", label="O"),
        LabelPrediction(token="not", label="O"),
        LabelPrediction(token="limited", label="O"),
        LabelPrediction(token="to", label="O"),
        LabelPrediction(token="a", label="O"),
        LabelPrediction(token="mistaken", label="O"),
        LabelPrediction(token="belief", label="O"),
        LabelPrediction(token="among", label="O"),
        LabelPrediction(token="consumers", label="O"),
        LabelPrediction(token="that", label="O"),
        LabelPrediction(token="the", label="O"),
        LabelPrediction(token="plaintiff", label="O"),
        LabelPrediction(token="is", label="O"),
        LabelPrediction(token="the", label="O"),
        LabelPrediction(token="producer", label="O"),
        LabelPrediction(token="of", label="O"),
        LabelPrediction(token="the", label="O"),
        LabelPrediction(token="defendant", label="O"),
        LabelPrediction(token="", label="O"),
        LabelPrediction(token="s", label="O"),
        LabelPrediction(token="goods", label="O"),
        LabelPrediction(token=".", label="O"),
        LabelPrediction(token="Star", label="B-CASE_NAME"),
        LabelPrediction(token="##bu", label="I-CASE_NAME"),
        LabelPrediction(token="##cks", label="I-CASE_NAME"),
        LabelPrediction(token="Corp", label="I-CASE_NAME"),
        LabelPrediction(token=".", label="I-CASE_NAME"),
        LabelPrediction(token="v", label="I-CASE_NAME"),
        LabelPrediction(token=".", label="I-CASE_NAME"),
        LabelPrediction(token="Wolfe", label="I-CASE_NAME"),
        LabelPrediction(
            token="'",
            label="I-CASE_NAME",
        ),
        LabelPrediction(token="s", label="I-CASE_NAME"),
        LabelPrediction(token="Borough", label="I-CASE_NAME"),
        LabelPrediction(token="Coffee", label="I-CASE_NAME"),
        LabelPrediction(token=",", label="I-CASE_NAME"),
        LabelPrediction(token="Inc", label="I-CASE_NAME"),
        LabelPrediction(token=".", label="I-CASE_NAME"),
        LabelPrediction(token=",", label="O"),
        LabelPrediction(token="58", label="B-VOLUME"),
        LabelPrediction(token="##8", label="I-VOLUME"),
        LabelPrediction(token="F", label="B-REPORTER"),
        LabelPrediction(token=".", label="I-REPORTER"),
        LabelPrediction(token="3", label="I-REPORTER"),
        LabelPrediction(token="##d", label="I-REPORTER"),
        LabelPrediction(token="97", label="B-PAGE"),
        LabelPrediction(token=",", label="O"),
        LabelPrediction(token="114", label="B-PIN"),
        LabelPrediction(token="(", label="O"),
        LabelPrediction(token="2d", label="B-COURT"),
        LabelPrediction(token="C", label="I-COURT"),
        LabelPrediction(token="##ir", label="I-COURT"),
        LabelPrediction(token=".", label="I-COURT"),
        LabelPrediction(token="2009", label="B-YEAR"),
        LabelPrediction(token=")", label="O"),
        LabelPrediction(token=".", label="O"),
        LabelPrediction(token="[SEP]", label="O"),
    ]
    ents = aggregate_entities(labels)
    res = CaselawCitation.from_token_label_pairs(ents)
    print(res)


@app.command()
def test_model():
    model = get_model()

    text = """A. The District Court Erred in Granting Summary Judgment on Appellant's Claim of Negligence\n The District Court's decision to grant summary judgment on the basis of lack of duty was in error. In Smith v. City of Los Angeles, 123 F.3d 456, 459 (9th Cir. 2000), this Court held that "a property owner owes a duty of care to entrants on their property, regardless of the nature of the activity the entrant is engaged in." Moreover, the California Civil Code § 2100 explicitly states that "all property owners owe entrants a basic level of care."
    """

    sentences = split_text(text)

    for s in sentences:
        msg.info(s)
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

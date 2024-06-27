import asyncio

import torch
import typer
from datasets import Dataset
from transformers import BertTokenizerFast

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
def test_model():
    device = torch.device("cuda")
    model = load_model_from_checkpoint()
    model = model.to(device)  # pyright: ignore
    # print(model)

    tokenizer = get_tokenizer()
    text = """warrants similar to the one at issue here. (Opp'n at 9-10.) See, e.g., United States v. Westley, No. 17-CR-171 (MPS), 2018 WL 3448161, at *12 (D. Conn. July 17, 2018) (in upholding Facebook warrant, equating Facebook information to other "electronic evidence" and asserting that "extremely broad" disclosure is a "practical necessity when dealing with electronic evidence");"""

    tokenized_input = tokenizer(text, return_tensors="pt", padding=True)
    tokenized_input = {k: v.to(device) for k, v in tokenized_input.items()}

    model.eval()

    with torch.no_grad():
        outputs = model(**tokenized_input)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)

    predicted_labels = [ALL_LABELS[p] for p in predictions[0].tolist()]
    tokens = tokenizer.convert_ids_to_tokens(tokenized_input["input_ids"][0])

    for token, label in zip(tokens, predicted_labels):
        print(f"{token} - {label}")


@app.command()
def push_to_hub():
    model = load_model_from_checkpoint()
    model.push_to_hub("ss108/legal-citation-bert", use_temp_dir=True)

    tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)
    tokenizer.push_to_hub("ss108/legal-citation-bert", use_temp_dir=True)


@app.command()
def hi():
    print("hi")


if __name__ == "__main__":
    app()

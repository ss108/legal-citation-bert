import asyncio

import torch
import typer
from datasets import Dataset
from transformers import BertTokenizerFast

from src.benchmarking.model import get_labels, get_model, split_text
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
from src.benchmarking.temp_aggregation import CaselawCitation

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



@app.command()
def test_model():
    model = get_model()

    text = """There are genuine issues of material fact as to whether the respondent’s use of his mark is likely to cause confusion among consumers. To prevail on a trademark infringement claim under the Lanham Act, a plaintiff must prove that the allegedly infringing mark would likely cause confusion among consumers. See 15 U.S.C. § 1114. “Confusion” in this context is not limited to a mistaken belief among consumers that the plaintiff is the producer of the defendant’s goods. Starbucks Corp. v. Wolfe’s Borough Coffee, Inc., 588 F.3d 97, 114 (2d Cir. 2009). The confusion could also be a mistaken belief that the plaintiff approves of the defendant’s use of the allegedly infringing mark and the underlying good or service to which it is applied. Dallas Cowboys Cheerleaders, Inc. v Pussycat Cinema, Ltd., 604 F.2d 200, 204 (2d Cir. 1979). Damage to the plaintiff’s reputation by association with the defendant and their trademark also constitutes confusion for the purposes of the Lanham Act. Id. at 205. 
    """

    sentences = split_text(text)

    for s in sentences:
        res = get_labels(s, model)
        print(res)


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

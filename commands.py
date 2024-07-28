import asyncio
import time
from typing import List

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
from src.data.generate import (
    Sentence,
    generate_prose_statute_citation,
    generate_tags,
    generate_unofficial_citation,
)
from src.data.prepare import (
    RAW_DATA_DIR,
    create_candidate_dataset,
    delete_from_cache,
    do_sentences,
    gather_wrapper,
    load_candidate_ds,
    load_for_training,
    load_raw_cl_docket_entries_ds,
    process_cl_doc,
    save_cl_docket_entries_ds,
    save_data_to_file,
    sents_to_data,
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
    # args = DataGenerationArgs(
    #     cit_form=CIT_FORM.SHORT, cit_type=CIT_TYPE.CASE, number=20
    # )
    # asyncio.run(do_sentences(args))  # pyright: ignore

    asyncio.run(gen_prose_statute_data())


async def gen_prose_statute_data():
    # res = await generate_prose_statute_citation(2)
    res: List[Sentence] = [
        {
            "text": "(§ 12940, subd. (j)(1); Carrisales v. Department of Corrections (1999) 21 Cal.4th 1132, 1136-1137 [90 Cal.Rptr.2d 804, 988 P.2d 1083].) In the",
        }
    ]
    data = await sents_to_data(res)
    file_name = f"{RAW_DATA_DIR}/prose_statutes_{int(time.time())}.jsonl"
    await save_data_to_file(data, file_name)


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
def prepare_candidate_dataset(version: str = "v0"):
    create_candidate_dataset(version)


@app.command()
def save_hf_ds(version: str = "v0"):
    ds = load_candidate_ds(version)
    split_and_save(ds, version)


@app.command()
def create_and_save_ds(version: str = "v0"):
    create_candidate_dataset(version)
    split_and_save_ds(version)
    # ds = load_candidate_ds(version)
    # split_and_save(ds, version)

@app.command()
def split_and_save_ds(version: str = "v0"):
    ds = load_candidate_ds(version)
    split_and_save(ds, version)

@app.command()
def train(version: str = "v0"):
    ds = load_for_training(version)
    # _, trainer = train_model(ds)
    # test_predict(trainer, ds["test"])


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
def test_model():
    model = get_model()

    text = """An employer's liability under FEHA for hostile environment sexual harassment committed by customers or clients prior to the effective date of the 2003 amendment to section 12940, subdivision (j) (Stats. 2003, ch. 671, § 1) is uncertain."""

    sentences = split_text(text)

    for s in sentences:
        res = get_labels(s, model)
        # print(res)
        f = citations_from(res)
        # print(f)


# @app.command()
# def test_model_agg():
#     model = get_model()

#     text = """ The confusion
# could also be a mistaken belief that the plaintiff approves of the defendant’s use of the allegedly infringing mark and the underlying good or service to which it is applied. Dallas Cowboys Cheerleaders, Inc. v Pussycat Cinema, Ltd., 604 F.2d 200, 204 (2d Cir. 1979). Damage to the
# plaintiff’s reputation by association with the defendant and"""

#     sentences = split_text(text)

#     for s in sentences:
#         res = get_labels(s, model)
#         cits = citations_from(res)
#         if len(cits) > 0:
#             print(cits)


@app.command()
def push_to_hub():
    model = load_model_from_checkpoint()
    model.push_to_hub("ss108/legal-citation-bert", use_temp_dir=True)  # pyright: ignore

    tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)
    tokenizer.push_to_hub("ss108/legal-citation-bert", use_temp_dir=True)  # pyright: ignore


if __name__ == "__main__":
    app()

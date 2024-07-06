from typing import Dict, List

import spacy
import torch
from transformers import AutoModelForTokenClassification, PreTrainedTokenizerFast

from src.training.model import (
    ALL_LABELS,
    get_tokenizer,
    load_model_from_checkpoint,
)

from .temp_aggregation import LabelPrediction

DEVICE = torch.device("cuda")


def get_model():
    model = load_model_from_checkpoint()
    model = model.to(DEVICE)  # pyright: ignore
    return model


def tokenize(
    s: str,
) -> Dict[str, torch.Tensor]:  # actually a HF BatchEncoding object, a subclass of dict
    # but that class isn't great to work with from a typing perspective
    tokenizer: PreTrainedTokenizerFast = get_tokenizer()
    tokenized_input = tokenizer(s, return_tensors="pt", padding=True)  # pyright: ignore
    tokenized_input: Dict[str, torch.Tensor] = {
        k: v.to(DEVICE) for k, v in tokenized_input.items()
    }
    return tokenized_input


def split_text(text: str) -> List[str]:
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)

    sentences: List[str] = [sent.text for sent in doc.sents]
    return sentences


def get_labels(
    text: str, model: AutoModelForTokenClassification
) -> List[LabelPrediction]:
    tokenizer: PreTrainedTokenizerFast = get_tokenizer()

    tokenized_input = tokenizer(text, return_tensors="pt", padding=True)  # pyright: ignore
    tokenized_input: Dict[str, torch.Tensor] = {
        k: v.to(DEVICE) for k, v in tokenized_input.items()
    }

    model.eval()  # pyright: ignore

    with torch.no_grad():
        outputs = model(**tokenized_input)  # pyright: ignore
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)

    predicted_labels = [ALL_LABELS[p] for p in predictions[0].tolist()]

    tokens = tokenizer.convert_ids_to_tokens(tokenized_input["input_ids"][0])  # pyright: ignore

    res = []
    for token, label in zip(tokens, predicted_labels):
        res.append(LabelPrediction(token=token, label=label))

    return res

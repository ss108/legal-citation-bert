from typing import Dict, List

import torch
import spacy
from transformers import AutoModelForTokenClassification, PreTrainedTokenizerFast
from src.training.model import (
    ALL_LABELS,
    get_tokenizer,
    # load_model_from_checkpoint,
)
from wasabi import msg


# Device detection logic
def _get_device() -> torch.device:
    """
    Detects if CUDA or MPS is available and returns the appropriate device.
    Defaults to CPU if neither is available.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        msg.info("CUDA is available; using GPU for inference.")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
        msg.info("MPS is available; using MPS for inference.")
    else:
        device = torch.device("cpu")
        msg.warn("No GPU found :(\nUsing CPU for inference.")
    return device


# Cache the detected device
DEVICE = _get_device()


def tokenize(s: str) -> Dict[str, torch.Tensor]:
    """
    Tokenizes the input string and moves the tensors to the appropriate device.
    """
    tokenizer: PreTrainedTokenizerFast = get_tokenizer()
    tokenized_input = tokenizer(s, return_tensors="pt", padding=True, truncation=True)  # pyright: ignore
    tokenized_input: Dict[str, torch.Tensor] = {
        k: v.to(DEVICE) for k, v in tokenized_input.items()
    }
    return tokenized_input


def split_text(text: str) -> List[str]:
    """
    Splits the input text into sentences using spaCy.
    """
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    sentences: List[str] = [sent.text for sent in doc.sents]
    return sentences


def get_labels(text: str, model: AutoModelForTokenClassification) -> List[str]:
    """
    Tokenizes the text, performs inference with the model, and returns predicted labels.
    """
    tokenizer: PreTrainedTokenizerFast = get_tokenizer()
    tokenized_input = tokenize(text)

    model.to(DEVICE)  # pyright: ignore
    model.eval()  # pyright: ignore

    with torch.no_grad():
        outputs = model(**tokenized_input)  # pyright: ignore
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)

    predicted_labels = [ALL_LABELS[p] for p in predictions[0].tolist()]
    tokens = tokenizer.convert_ids_to_tokens(tokenized_input["input_ids"][0])  # pyright: ignore

    res = []
    raw_pairs = []
    for token, label in zip(tokens, predicted_labels):
        raw_pairs.append((token, label))
        res.append(LabelPrediction(token=token, label=label))

    # Log the token-label pairs for debugging
    msg.info(f"Token-label pairs: {raw_pairs}")

    return res

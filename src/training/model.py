from pathlib import Path
from typing import Optional

import torch
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    PreTrainedTokenizerFast,
)
from wasabi import msg
from src.training.constants import ALL_LABELS, MODEL_NAME
import spacy


def get_base_model():
    # Load the configuration from the pre-trained model
    config = AutoConfig.from_pretrained(MODEL_NAME)

    # Create the model with the original configuration
    model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME, config=config)

    # Modify the classifier to match the number of your labels
    model.classifier = torch.nn.Linear(config.hidden_size, len(ALL_LABELS))
    model.num_labels = len(ALL_LABELS)

    return model


def get_tokenizer() -> PreTrainedTokenizerFast:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    assert isinstance(tokenizer, PreTrainedTokenizerFast)
    return tokenizer


def load_model_from_checkpoint(
    version: Optional[str] = None,
) -> AutoModelForTokenClassification:
    training_output_dir = (
        Path(__file__).resolve().parent.parent.parent / "training_output"
    )

    if not training_output_dir.exists():
        raise FileNotFoundError(
            f"Training output directory {training_output_dir} not found."
        )

    if version:
        checkpoint_dir = training_output_dir / f"v{version}"
        if not checkpoint_dir.exists():
            raise FileNotFoundError(
                f"No checkpoint directory found for version {version}"
            )

    else:
        checkpoint_dir = max(
            training_output_dir.glob("v*"),
            key=lambda x: x.stat().st_mtime,
            default=None,
        )
        if checkpoint_dir is None:
            raise FileNotFoundError(
                "No checkpoint directories found in the training output directory."
            )

    msg.info(f"Loading model from checkpoint directory {checkpoint_dir}")

    # Find the checkpoint within the selected directory
    checkpoint_path = next(checkpoint_dir.glob("checkpoint-*"), None)
    if checkpoint_path is None:
        raise FileNotFoundError("No checkpoint found in the specified directory")

    config = AutoConfig.from_pretrained(checkpoint_path, num_labels=len(ALL_LABELS))

    # Load the model from the checkpoint directory with the correct configuration
    model = AutoModelForTokenClassification.from_pretrained(
        checkpoint_path, config=config
    )

    return model


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


def tokenize(s: str) -> dict[str, torch.Tensor]:
    """
    Tokenizes the input string and moves the tensors to the appropriate device.
    """
    tokenizer: PreTrainedTokenizerFast = get_tokenizer()
    tokenized_input = tokenizer(s, return_tensors="pt", padding=True, truncation=True)  # pyright: ignore
    tokenized_input: dict[str, torch.Tensor] = {
        k: v.to(DEVICE) for k, v in tokenized_input.items()
    }
    return tokenized_input


def split_text(text: str) -> list[str]:
    """
    Splits the input text into sentences using spaCy.
    """
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    sentences: list[str] = [sent.text for sent in doc.sents]
    return sentences


def get_labels(text: str, model: AutoModelForTokenClassification) -> list[str]:
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
    for token, label in zip(tokens, predicted_labels):
        res.append(f"Token: {token}, Label: {label}")

    return res

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

MODEL_NAME = "dslim/bert-base-NER"

CASE_LABELS = [
    "B-CASE_NAME",
    "I-CASE_NAME",
    "B-VOLUME",
    "I-VOLUME",
    "B-REPORTER",
    "I-REPORTER",
    "B-PAGE",
    "I-PAGE",
    "B-PIN",
    "I-PIN",
    "B-COURT",
    "I-COURT",
    "B-YEAR",
    "I-YEAR",
]

STAT_LABELS = [
    "B-TITLE",
    "I-TITLE",
    "B-CODE",
    "I-CODE",
    "B-SECTION",
    "I-SECTION",
]

SHORT_LABELS = ["B-ID", "I-ID", "B-SUPRA", "I-SUPRA"]


ALL_LABELS = CASE_LABELS + STAT_LABELS + SHORT_LABELS + ["O"]

LABEL_MAP = {label: i for i, label in enumerate(ALL_LABELS)}


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

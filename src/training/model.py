from pathlib import Path
from typing import Optional

import torch
from transformers import AutoConfig, AutoModelForTokenClassification, AutoTokenizer

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


def get_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    return tokenizer


def load_model_from_checkpoint(timestamp: Optional[str] = None):
    """
    Load a trained model from the most recent checkpoint or a specified checkpoint directory,
    with the number of labels it was trained with.

    Args:
    timestamp (str, optional): Timestamp of the training run to identify the checkpoint directory.

    Returns:
    model: The loaded pretrained model configured with the correct number of labels.
    """
    training_output_dir = (
        Path(__file__).resolve().parent.parent.parent / "training_output"
    )

    # Ensure the base training output directory exists
    if not training_output_dir.exists():
        raise FileNotFoundError(
            f"Training output directory {training_output_dir} not found."
        )

    # If a timestamp is provided, use it to determine the checkpoint directory
    if timestamp:
        checkpoint_dir = training_output_dir / f"run_{timestamp}"
        if not checkpoint_dir.exists():
            raise FileNotFoundError(
                f"No checkpoint directory found for timestamp {timestamp}"
            )
    else:
        # If no timestamp is provided, find the most recent checkpoint directory
        checkpoint_dir = max(
            training_output_dir.glob("run_*"),
            key=lambda x: x.stat().st_mtime,
            default=None,
        )
        if checkpoint_dir is None:
            raise FileNotFoundError(
                "No checkpoint directories found in the training output directory."
            )

    # Find the checkpoint within the selected directory
    checkpoint_path = next(checkpoint_dir.glob("checkpoint-*"), None)
    if checkpoint_path is None:
        raise FileNotFoundError("No checkpoint found in the specified directory")

    # Load the configuration from the checkpoint, ensuring it matches expected num_labels
    # Assume ALL_LABELS is defined globally or passed correctly
    config = AutoConfig.from_pretrained(checkpoint_path, num_labels=len(ALL_LABELS))

    # Load the model from the checkpoint directory with the correct configuration
    model = AutoModelForTokenClassification.from_pretrained(
        checkpoint_path, config=config
    )

    return model

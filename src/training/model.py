from pathlib import Path
from typing import Optional

# import torch
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    PreTrainedTokenizerFast,
)
from wasabi import msg
from src.training.constants import ALL_LABELS, MODEL_NAME


# def get_base_model():
#     # Load the configuration from the pre-trained model
#     config = AutoConfig.from_pretrained(MODEL_NAME)

#     # Create the model with the original configuration
#     model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME, config=config)

#     # Modify the classifier to match the number of your labels
#     model.classifier = torch.nn.Linear(config.hidden_size, len(ALL_LABELS))
#     model.num_labels = len(ALL_LABELS)

#     return model


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

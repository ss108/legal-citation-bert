from pathlib import Path

import numpy as np
import torch
from datasets import Dataset, DatasetDict
from transformers import EarlyStoppingCallback, Trainer, TrainingArguments

from .model import get_base_model

OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "training_output"


def get_next_version() -> str:
    version_dirs = [
        d for d in OUTPUT_DIR.iterdir() if d.is_dir() and d.name.startswith("v")
    ]

    version_numbers = sorted(
        [int(d.name[1:]) for d in version_dirs if d.name[1:].isdigit()]
    )

    next_version_number = version_numbers[-1] + 1 if version_numbers else 1

    next_version_dir = f"v{next_version_number}"

    return next_version_dir


def get_output_dir():
    output_dir = OUTPUT_DIR / get_next_version()

    output_dir.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
    return output_dir


def count_errors(predictions):
    pred_labels = np.argmax(predictions.predictions, axis=2)
    error_count = 0
    valid_tokens_count = 0  # Total number of valid tokens that were compared

    for i in range(len(predictions.label_ids)):
        relevant_indices = np.where(predictions.label_ids[i] != -100)  # Ignore padding
        errors = np.sum(
            pred_labels[i][relevant_indices]
            != predictions.label_ids[i][relevant_indices]
        )
        error_count += errors
        valid_tokens_count += len(predictions.label_ids[i][relevant_indices])

    error_percentage = (
        (error_count / valid_tokens_count) * 100 if valid_tokens_count > 0 else 0
    )
    return {"total_errors": error_count, "error_percentage": error_percentage}


def train_model(ds_dict: DatasetDict):
    if torch.cuda.is_available():
        device_index = torch.cuda.current_device()  # Get current device index
        torch.cuda.set_device(device_index)  # Set the default CUDA device by index
        print(f"Using GPU: {torch.cuda.get_device_name(device_index)}")
    else:
        raise EnvironmentError(
            "No GPU found! Make sure the training is supposed to run on a GPU."
        )

    model = get_base_model()

    output_dir = get_output_dir()

    print(f"Training output directory: {output_dir}")

    # Define training arguments with adjusted batch sizes and number of epochs
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=75,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=1,
        warmup_steps=10,
        weight_decay=0.01,
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,
        do_train=True,
        do_eval=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        save_total_limit=1,  # Optionally keep all checkpoints from this run
        fp16=torch.cuda.is_available(),
    )

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds_dict["train"],
        eval_dataset=ds_dict["valid"],
        compute_metrics=None,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    # Train the model
    trainer.train()

    return model, trainer


def test_predict(trainer, test_data: Dataset):
    # Predict on the test dataset
    predictions = trainer.predict(test_data)

    error_results = count_errors(predictions)

    print(error_results)

    return predictions.predictions, error_results

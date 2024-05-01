from collections import Counter

import torch
from datasets import Dataset, load_metric
from torch.nn import Softmax
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)

from src.data.prepare import get_formatted_dataset
from src.training.model import MODEL_NAME, get_base_model

dataset = get_formatted_dataset("v1")

test_data = dataset["test"]


softmax = Softmax(dim=1)


class Predictor:
    model: AutoModelForSequenceClassification
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast
    device: torch.device

    def __init__(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = get_base_model().to(device)
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def split_text(self, text):
        tokens = self.tokenizer.tokenize(text)
        max_length = self.tokenizer.model_max_length - 2

        return [
            self.tokenizer.convert_tokens_to_string(tokens[i : i + max_length])
            for i in range(0, len(tokens), max_length)
        ]

    def predict_chunk(self, chunk):
        inputs = self.tokenizer(chunk, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)  # pyright: ignore
        probabilities = softmax(outputs.logits)
        predicted_class = torch.argmax(probabilities).item()
        return predicted_class

    def predict_text(self, text):
        chunks = self.split_text(text)
        predictions = [self.predict_chunk(chunk) for chunk in chunks]
        # Use Counter to find the most common element (mode)
        if predictions:
            most_common = Counter(predictions).most_common(1)
            mode_result = most_common[0][0]  # The most common prediction
            mode_count = most_common[0][1]  # How many times it appears
        else:
            mode_result = None  # Handle the case where there are no predictions
            mode_count = 0

        # print(f"Predictions: {predictions}")
        # print(f"Mode result: {mode_result}, Count: {mode_count}")

        return (
            mode_result if mode_result is not None else 0
        )  # Default or error handling return

    def get_test_labels(self):
        test_texts = [
            "The law, per section 56, mandates a 35-hour workweek.",  # Clearly 'law'
            "Water boils at 100 degrees Celsius.",  # Clearly 'fact'
        ]

        predictions = []

        for t in test_texts:
            prediction = self.predict_text(t)  # pyright: ignore
            predictions.append(prediction)

        return {"law": predictions[0], "fact": predictions[1]}


# def run_on_ds(dataset: Dataset):
#     max_length = tokenizer.model_max_length - 2  # Subtract 2 for the special tokens

#     predictions = []
#     references = []

#     for row in dataset:
#         prediction = predict_text(model, tokenizer, row["text"], max_length)  # pyright: ignore
#         predictions.append(prediction)

#     return predictions


# def calc_accuracy(predictions):
#     metric = load_metric("accuracy")

#     # Calculate the accuracy
#     accuracy = metric.compute(predictions=predictions, references=test_data["type"])  # pyright: ignore
#     print(f"Baseline accuracy: {accuracy}")


# def get_baseline():
#     # print(test_data)
#     predictions = run_on_ds(test_data)
#     calc_accuracy(predictions)


# def tokenize_fn(row):
#     return tokenizer(row["text"], padding="max_length", truncation=False)


# def find_max_token_length():
#     max_length = 0
#     for row in test_data:
#         tokens = tokenizer(row["text"], add_special_tokens=True)  # pyright: ignore
#         max_length = max(max_length, len(tokens["input_ids"]))  # pyright: ignore
#     return max_length


# max_length = tokenizer.model_max_length - 2  # Subtract 2 for the special tokens

# predictions = []
# for row in test_data:
#     prediction = predict_text(model, tokenizer, row["text"], max_length)  # pyright: ignore
#     predictions.append(prediction)

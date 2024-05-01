# type: ignore

from datasets import load_dataset

# scotus_filings = load_dataset("pile-of-law/pile-of-law", "scotus_filings")
# courtListener_opinions = load_dataset(
#     "pile-of-law/pile-of-law", "courtListener_opinions"
# )
dataset = load_dataset(
    "pile-of-law/pile-of-law", "courtlistener_docket_entry_documents"
)


def inspect_cl():
    print(dataset.column_names)  # pyright: ignore
    # print(courtListener_docket_entry_documents["train"][:3])  # pyright:
    # ignore
    num_train_rows = dataset["train"].num_rows

    num_validation_rows = dataset["validation"].num_rows

    print(f"Number of training rows: {num_train_rows}")
    print(f"Number of validation rows: {num_validation_rows}")

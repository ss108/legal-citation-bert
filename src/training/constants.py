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

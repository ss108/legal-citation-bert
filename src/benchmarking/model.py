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

import os
import re
from typing import Dict, List

import nltk


def remove_multiple_newlines(text: str) -> str:
    return re.sub(r"\n{2,}", "\n", text)


def clean_text(text: str):
    text = remove_multiple_newlines(text)
    text = re.sub(r"\d+\s+of\s+\d+", "", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def clean_nyscef(text: str) -> str:
    """
    Remove all the NYSCEF cruft and boilerplate and crap from the text.
    """
    efiling_pattern = r"FILED: [A-Z]+\sCOUNTY\sCLERK[\s\S]*?NYSCEF DOC\. NO\. \d+ RECEIVED NYSCEF: \d+/\d+/\d+"

    return re.sub(efiling_pattern, "", text, flags=re.DOTALL)


def extract_nyscef_metadata(text: str) -> Dict:
    doc_number_pattern = r"NYSCEF\sDOC.\sNO.\s([\d]+)\s"
    m = re.search(doc_number_pattern, text)

    if m:
        return {"nyscef_doc_num": int(m.groups(1)[0])}

    return {}


def get_text(file_path: str) -> str:
    with open(file_path, "r") as file:
        file_content = file.read()

    return file_content


def get_dir_files(dir: str) -> List[str]:
    dir_path = f"./data_files/{dir}"
    dir_files: List[str] = os.listdir(dir_path)
    file_texts: List[str] = [get_text(f"{dir_path}/{f}") for f in dir_files]
    return [clean_text(clean_nyscef(t)) for t in file_texts]


def get_fact_sections() -> List[str]:
    return get_dir_files("facts")


def get_law_sections() -> List[str]:
    return get_dir_files("law")


def split_sentences(text: str) -> List[str]:
    sentences = nltk.sent_tokenize(text)
    return sentences

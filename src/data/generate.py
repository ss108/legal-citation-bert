import asyncio
import json
from typing import List, Optional, Tuple, TypedDict, get_type_hints

from src.data.types import CIT_FORM, CIT_TYPE, DataGenerationArgs, Sentence
from src.openai import chat
from src.training.constants import ALL_LABELS

sentence_schema = get_type_hints(Sentence)


EXAMPLES_MAP_CASE = {
    CIT_FORM.LONG: """
    The output for this run should be long, full case citations.\n
    EXAMPLES:
    Text: "This was said in People v. Jones, 123 F. 3d 456 (S.D.N.Y. 1996)."\n
    Text: "Farx v. Passadoe, 123 U.S. 457, 458. And therefore, plaintiff
    requests 50 billion dollars."
    (1987)."\n
    Text: "Have you ever seen the rain? Nils v. B, 34 A.D. 87, 90 (N.J. 2023)
    (querying whether rain has ever been truly seen)."\n
    Text: "Defendant's conduct is similar to that seen in Wahl v. Wahl, 123 34
    N.Y. 2d 45, 46 (N.Y. App. Div. 1999)."\n
    """,
    CIT_FORM.SHORT: """
    The output for this run should be short-form caselaw citations.\n
    EXAMPLES:
    Text: "This was said in Vig, 123 F.3d at 89."\n
    Text: "HELLO, THE LAW SAYS GOOD. Karlsburg, 129 U.S. at 458. And so, we ask, good?"\n
    Text: "As stated therein. Bryer Corp., 34 A.D. at 90. More words follow."\n
    """,
}

STATUTE_EXAMPLE = """
    The output for this run should be long, full statute citations.\n
    EXAMPLES:
    Text: "The law says this. 28 U.S.C. § 1234 (1999)."\n
    Text: "This was contratry to N.Y. CPLR § 787."\n
    """

EXAMPLES_MAP_STATUTE = {
    CIT_FORM.LONG: STATUTE_EXAMPLE,
    CIT_FORM.SHORT: STATUTE_EXAMPLE,
}


async def generate(args: DataGenerationArgs) -> List[Sentence]:
    if args.cit_form == CIT_FORM.SHORT:
        return await _generate_short(args.number, args.cit_type)
    else:
        return await _generate_long(args.number, args.cit_type)


async def _generate_long(n: int, type: CIT_TYPE = CIT_TYPE.CASE) -> List[Sentence]:
    examples_map = EXAMPLES_MAP_CASE if type == CIT_TYPE.CASE else EXAMPLES_MAP_STATUTE

    PROMPT = f"""
    GOAL: Generate a snippet from a fictitious legal motion containing an
    American legal citation. The sentence will be used to train a NER model.

    The jurisdiction could be Federal or state, appellate or not.
    \n
    {examples_map[CIT_FORM.LONG]}
    OUTPUT in this JSON format: {sentence_schema}
    """
    tasks = []
    for _ in range(n):
        tasks.append(
            chat(
                messages=[],
                system_prompt=PROMPT,
                temperature=1.0,
                model="gpt-4-turbo-preview",
            )
        )

    raw_results = await asyncio.gather(*tasks)
    formatted_results = []

    for r in raw_results:
        try:
            sent = json.loads(r)
            formatted_results.append(sent)
        except Exception:
            continue

    return formatted_results


async def _generate_short(n: int, type: CIT_TYPE = CIT_TYPE.CASE) -> List[Sentence]:
    examples_map = EXAMPLES_MAP_CASE if type == CIT_TYPE.CASE else EXAMPLES_MAP_STATUTE

    PROMPT = f"""
    GOAL: Generate a snippet from a fictitious legal motion containing an
    American legal citation. The sentence will be used to train a NER model.

    \n
    {examples_map[CIT_FORM.SHORT]}
    OUTPUT in this JSON format: {sentence_schema}
    """
    tasks = []
    for _ in range(n):
        tasks.append(
            chat(
                messages=[],
                system_prompt=PROMPT,
                temperature=1.0,
                model="gpt-4-turbo-preview",
            )
        )

    raw_results = await asyncio.gather(*tasks)
    formatted_results = []

    for r in raw_results:
        try:
            sent = json.loads(r)
            formatted_results.append(sent)
        except Exception:
            continue

    return formatted_results


async def generate_unofficial_citation(n: int = 1) -> List[Sentence]:
    PROMPT = f"""
    GOAL: Generate a snippet from a fictitious legal motion containing a caselaw
    citation to the Lexis or Westlaw reporter, or with no reporter at all and
    only a court case number. The sentence will be used to train a NER model.

    The jurisdiction could be Federal or state, appellate or not.
    \n
    Example of Lexis: Byuk v. Sung, 2024 Lexis 99873 (S.D.N.Y. Mar. 11, 2024)\n
    Example of Westlaw: United States v. Westley, 2018 WL 3448161 at *8 (D.
    Conn. July 17, 2018)\n
    Example of no reporter: People v. Jones, No. 17-CR-171 (MPS) at *99 (D. Conn. July 11th, 2018)\n
    OUTPUT in this JSON format: {sentence_schema}
    """
    tasks = []
    for _ in range(n):
        tasks.append(
            chat(
                messages=[],
                system_prompt=PROMPT,
                temperature=1.0,
                model="gpt-4-turbo-preview",
            )
        )

    raw_results = await asyncio.gather(*tasks)
    formatted_results = []

    for r in raw_results:
        try:
            sent = json.loads(r)
            formatted_results.append(sent)
        except Exception:
            continue

    return formatted_results


async def generate_prose_statute_citation(n: int = 1) -> List[Sentence]:
    PROMPT = f"""
    GOAL: Generate a snippet from a fictitious legal motion containing a statute which is cited not per the Bluebook, but in a more prose-like manner. The sentence will be used to train a NER model.

    The 'statute' should be a citable legal provision, that furnishes support for typical arguments, as opposed to a reference to a legislative act.

    EXAMPLES:\n
    - Per Section 870 of the CPLR...
    - Plaintiff brings this complaint pursuant to Government Code Section 999(b)(1)(ii).
    - Accordingly, the defendant's motion thus fails to comply with the requirements of Title 28, and should thus be denied.
    - The court finds that the defendant's actions were in violation of the California Penal Code, particularly Section 1234(a).
    - This has many subsections: see  U.S.C., Title 28, section 1234(a)(3)(b).

    OUTPUT in this JSON format: {sentence_schema}
    """

    tasks = []
    for _ in range(n):
        tasks.append(
            chat(
                messages=[],
                system_prompt=PROMPT,
                temperature=1.0,
                model="gpt-4o",
            )
        )

    raw_results = await asyncio.gather(*tasks)  # type: ignore
    formatted_results: List[Sentence] = []

    for r in raw_results:
        try:
            assert len(r) > 0
            sent = json.loads(r)
            formatted_results.append(sent)
        except Exception:
            continue

    return formatted_results


class TokenTags(TypedDict):
    tags: List[Tuple[str, str]]


async def generate_tags(text: str, tokens: List[str]) -> Optional[TokenTags]:
    PROMPT = f"""
    The goal of the project is to train the BERT model to perform NER to extract
    legal citations.
    Here is some text, and its tokenized form from a BERT model.
    Given the following tags, return an array of tags/labels representing the
    appropriate tag for each token in the tokenized text.
    {ALL_LABELS} \n
    Here is a general overview of how the tags are to be used:
    For Caselaw:
    CASE_NAME: Includes the names of both parties, plus the 'v.', and all other
    punctuation, e.g. 'People v. Jones'. Each token from 'People', 'v.'
    (including the period), and 'Jones' is tagged as 'B-CASE_NAME' or 'I-CASE_NAME'.
    VOLUME: The volume number of the reporter, e.g. '123'.
    REPORTER: The reporter abbreviation, e.g. 'F. 3d'.
    PAGE: The page number of the reporter on which the case starts, e.g. '456'.
    PIN: Specific page citation, if present, e.g. 459. There might not be a pin
    cite. Remember that PIN always follows PAGE.
    COURT: The court that issued the opinion, e.g. 'S.D.N.Y.'.
    YEAR: The year the opinion was issued, e.g. 1999

    For Statutes:
    TITLE: The title of the statute, e.g. '28' in '28 CA. Code. Reg'.
    CODE: The abbreviation or name of the code, e.g. 'U.S.C.' or 'Fed. R. Civ. P.'
    SECTION: All sections or subsections of the statute. E.g. '1234' in '28
    U.S.C. 1234', or '12(a)(3)' in 'Fed. R. Civ. P. 12(a)(3)'.
    Some of the citations may be written out instead of abbreviated per the Bluebook; for example, instead of 12 U.S.C. § 1234, the citation may be written as 'Section 1234 of Title 12 of the United States Code'.

    Misc:
    ID: Used to label 'id' when it refers to the previous citation, e.g. Id. at
    87.
    SUPRA: Used for 'supra' citations, e.g. Supra at 99.
    \n
    FULL EXAMPLE: Text: 'Wowowowh. People v. Jones, 123 F. 3d 456 (S.D.N.Y. 1996)' \n
    Tokens: ["Wow", "##owo", "##wh", ".", "People", "v", ".", "Jones", ",",
    "123", "F.", "3", "##d", "456", "(", "S", ".", "D", ".", "N", ".", "Y", ".",
    "1996", ")"]
    should be tagged as:
    Tags:  [
    ("Wow", "O"),
    ("##owo", "O"),
    ("##wh", "O"),
    (".", "O"),
    ("People", "B-CASE_NAME"),
    ("v", "I-CASE_NAME"),
    (".", "I-CASE_NAME"),
    ("Jones", "I-CASE_NAME"),
    (",", "O"),
    ("123", "B-VOLUME"),
    ("F.", "B-REPORTER"),
    ("3", "I-REPORTER"),
    ("##d", "I-REPORTER"),
    ("456", "B-PAGE"),
    ("(", "O"),
    ("S", "B-COURT"),
    (".", "I-COURT"),
    ("D", "I-COURT"),
    (".", "I-COURT"),
    ("N", "I-COURT"),
    (".", "I-COURT"),
    ("Y", "I-COURT"),
    (".", "I-COURT"),
    ("1996", "B-YEAR"),
    (")", "O")
]
    \n
    FULL EXAMPLE 2: Text: 'Wexler & Greene, LLC v. Lachs, 250 Cal. Rptr. 3d 176, 180 (Cal. Ct. App. 2008).' \n
    Tokens: ['We', '##x', '##ler', '&', 'Greene', ',', 'LLC', 'v', '.', 'Lac', '##hs', ',', '250', 'Cal', '.', 'R', '##pt', '##r', '.', '3', '##d', '176', ',', '180', '(', 'Cal', '.', 'C', '##t', '.', 'A', '##pp', '.', '2008', ')']
    should be tagged as:
    Tags: [
    ("W", "B-CASE_NAME"),
    ("##x", "I-CASE_NAME"),
    ("##ler", "I-CASE_NAME"),
    ("&", "I-CASE_NAME"),
    ("Greene", "I-CASE_NAME"),
    (",", "I-CASE_NAME"),
    ("LLC", "I-CASE_NAME"),
    ("v", "I-CASE_NAME"),
    (".", "I-CASE_NAME"),
    ("Lac", "I-CASE_NAME"),
    ("##hs", "I-CASE_NAME"),
    (",", "O"),
    ("250", "B-VOLUME"),
    ("Cal", "B-REPORTER"),
    (".", "I-REPORTER"),
    ("R", "I-REPORTER"),
    ("##pt", "I-REPORTER"),
    ("##r", "I-REPORTER"),
    (".", "I-REPORTER"),
    ("3", "I-REPORTER"),
    ("##d", "I-REPORTER"),
    ("176", "B-PAGE"),
    (",", "O"),
    ("180", "B-PIN"),
    ("(", "O"),
    ("Cal", "B-COURT"),
    (".", "I-COURT"),
    ("C", "I-COURT"),
    ("##t", "I-COURT"),
    (".", "I-COURT"),
    ("A", "I-COURT"),
    ("##pp", "I-COURT"),
    (".", "I-COURT"),
    ("2008", "B-YEAR"),
    (")", "O")
]
    \n
    FULL EXAMPLE 3:
    Text: 'He loves crime. Id. at 99. Moreover,'
    Tags:[('He', 'O'), ('loves', 'O'), ('cri', 'O'), ('##me', 'O'), ('.', 'O'),
    ('Id', 'ID'), ('.', 'O'), ('at', 'O'), ('99', 'B-PIN'), ('.', 'O'), ('More',
    'O'), ('##over', 'O'), (',', 'O')]
    \n
    FULL EXAMPLE 4:
    Text: 'With the proper intent per 28 U.S.C. § 1234, the court'
    Tags: [('With', 'O'), ('the', 'O'), ('proper', 'O'), ('intent', 'O'),
    ('per', 'O'), ('28', 'B-TITLE'), ('U', 'B-CODE'), ('.', 'I-CODE'),
    ('S', 'I-CODE'), ('.', 'I-CODE'), ('C', 'I-CODE'), ('.', 'I-CODE'),
    ('§', 'O'), ('1234', 'B-SECTION'), (',', 'O'), ('the', 'O'), ('court', 'O')

    FULL EXAMPLE 5:
    Text: 'Counsel moves the court pursuant to Fed. R. Civ. P. 12(a)(3)'
    Tags: [('Counsel', 'O'), ('moves', 'O'), ('the', 'O'), ('court', 'O'),
    ('pursuant', 'O'), ('to', 'O'), ('Fed', 'B-CODE'), ('.', 'I-CODE'), ('R',
    'I-CODE'), ('.', 'I-CODE'), ('Civ', 'I-CODE'), ('.', 'I-CODE'), ('P',
    'I-CODE'), ('.', 'I-CODE'), ('12', 'B-SECTION'), ('(', 'I-SECTION'), ('a',
    'I-SECTION'), (')', 'I-SECTION'), ('(', 'I-SECTION'), ('3', 'I-SECTION'),
    (')' 'I-SECTION'),]

    FULL EXAMPLE 6:
    Text: 'industry. See 42 U.S .C. § 12201(c).'
    Tags: [('industry', 'O'), ('.', 'O'), ('See', 'O'), ('42', 'B-TITLE'), ('U', 'B-CODE'), ('.', 'I-CODE'), ('S', 'I-CODE'), ('.', 'I-CODE'), ('C', 'I-CODE'), ('.', 'I-CODE'), ('§', 'O'), ('12201', 'B-SECTION'), ('(', 'I-SECTION'), ('c', 'I-SECTION'), (')', 'I-SECTION')]

    FULL EXAMPLE 7:
    Text: An employer's liability under FEHA for hostile environment sexual harassment committed by customers or clients prior to the effective date of the 2003 amendment to section 12940, subdivision (j) (Stats. 2003, ch. 671, § 1) is uncertain.
    Tags: [('[CLS]', 'O'), ('An', 'O'), ('employer', 'O'), ("'", 'O'), ('s', 'O'), ('liability', 'O'), ('under', 'O'), ('F', 'O'), ('##E', 'O'), ('##HA', 'O'), ('for', 'O'), ('hostile', 'O'), ('environment', 'O'), ('sexual', 'O'), ('harassment', 'O'), ('committed', 'O'), ('by', 'O'), ('customers', 'O'), ('or', 'O'), ('clients', 'O'), ('prior', 'O'), ('to', 'O'), ('the', 'O'), ('effective', 'O'), ('date', 'O'), ('of', 'O'), ('the', 'O'), ('2003', 'O'), ('amendment', 'O'), ('to', 'O'), ('section', 'B-TITLE'), ('129', 'B-SECTION'), ('##40', 'I-SECTION'), (',', 'O'), ('subdivision', 'O'), ('(', 'I-SECTION'), ('j', 'I-SECTION'), (')', 'I-SECTION'), ('(', 'O'), ('St', 'O'), ('##ats', 'O'), ('.', 'O'), ('2003', 'O'), (',', 'O'), ('ch', 'O'), ('.', 'O'), ('67', 'O'), ('##1', 'O'), (',', 'O'), ('§', 'O'), ('1', 'O'), (')', 'O'), ('is', 'O'), ('uncertain', 'O'), ('.', 'O'), ('[SEP]', 'O')]

    \n
    Do not assign any labels not in the list above. If a citation is not for
    caselaw or statute, ignore it (assign its tokens 'O'). Ignore legislative
    materials/records, books, etc.
    \n
    Output your response in the following JSON format:
    {get_type_hints(TokenTags)}
    \n
    'tags' should be a list of tuples, where each tuple is a token and its label\n
    e.g. {{'tags': [('Wow', 'O'), ('ff', 'O'), ('word', 'O'), ('good', 'O'),
    ('People', 'B-CASE_NAME'), ('v', 'I-CASE_NAME'), ('.', 'I-CASE_NAME'),
    ('Stahl', 'I-CASE_NAME'), (',', 'O'), ('123', 'B-VOLUME'), ('F.',
    'B-REPORTER'), ('3', 'I-REPORTER'), ('##d', 'I-REPORTER'), ('456',
    'B-PAGE')...]}}
    \n
    Here are the text and tokens for your labeling task:
    Text: {text}
    Tokens: {tokens}
    """

    r = await chat(messages=[], system_prompt=PROMPT)

    try:
        return json.loads(r)
    except Exception:
        return None

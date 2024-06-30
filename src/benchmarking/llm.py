from __future__ import annotations

import json
from typing import Dict, List

import tiktoken

from src.benchmarking.types import CitationExtractionResult
from src.openai import chat

LLM_EXTRACTION_PROMPT = """
Given a block of text and a JSON object representing citations extracted from
the document thus far, add citations to the JSON object or increment the counts
as needed.

Output in the provided JSON format.

Additional rules/instructions:
- Ignore any citations to anything other than caselaw or statutes.
- If a case citation has multiple reporters, use the first one.
- If a caselaw citation lacks a name, use an empty string for the name. 
- Always return a value for both keys, even if the value is an empty dict.
- Count rules of civil procedure as statutes.
- Do not include pincites and parentheticals (format as if for a Table of Authorities)

EXAMPLE:
Input Object: {{}}
Input Text: The purpose of a motion to dismiss pursuant to Rule 12(b)(6) is to test
the legal sufficiency of the complaint. N. Star Int'l v. Ariz. Corp. Comm'n ,
720 F.2d 578, 581 (9th Cir. 1983). Also, a statute. 18 U.S.C. § 1961(1).

Output: {{'cases': {{'N. Star Int'l v. Ariz. Corp. Comm'n, 720 F.2d 578 (9th Cir. 1983)': 1}},
'statutes': {{'18 U.S.C. § 1961(1)': 1}}}}

EXAMPLE 2:
Input Object: {{'cases': {{'Some Other v. Case, 720 F.2d 578
(9th Cir. 1983)': 1}}, 'statutes': {{'26 U.S.C. § 172': 1}}}}
Input Text: Hi there. 26 U.S.C. § 172(b). Plaintiff contends it was improper for the
IRS to do this since this court invalidated that regulation. American Standard
v. United States, 220 Ct. Cl. 411, 602 F. 2d 256 (1979), rehearing en banc
denied (Oct. 12, 1979), and Union Carbide v. United States, 612
F. 2d 558 (1979). Plaintiff claims the methods indorsed by these cases, the
aggregate method with losses and the fractional method with losses, are the
proper methods for calculating the deduction and result in an increase in
Allied’s tax refund for 1971. 26 U.S.C. § 172. 

Output: 
{{'cases': {{'Some Other v. Case, 720 F.2d 578 (9th Cir. 1983)': 1, 'American
Standard v. United States, 220 Ct. Cl. 411 (1979)': 1, 'Union Carbide v. United
States, 612 F. 2d 558 (1979)': 1}}, 'statutes': {{'26 U.S.C. § 172': 2, '26 U.S.C.
§ 172(b)': 1}}}}

EXAMPLE 3:
Input Object: {{'cases': {{'Heck v. Jones, 67 F.3d 87 (3rd Cir. 1999)': 1}}, 'statutes': {{}}}}
Input Text: it was an intergovernmental task force composed of several local, county,
and state governmental entities, rather than a separate legal entity unto
itself. 65 F.3d 784, 791-92 (9th Cir. 1995). Because this case includes claims
against a cow, my life is rendered null. Jones, 67 F.3d at 89.

Output: {{'cases': {{'65 F.3d 784 (9th Cir. 1995)': 1, 'Heck v. Jones, 67 F.3d 87
(3rd Cir. 1999)': 2}}, 'statutes': {{}}}}

EXAMPLE 4:
Input Object: {{'cases': {{}}, 'statutes': {{'18 U.S.C. § 1961(1)': 1}}}}

Input Text: Because of 18 U.S.C. § 1961(1). (Doc. No. 12 at 11) (quoting Gonzalez v.
City of Anaheim , 747 F.3d 789, 795 (9th Cir. 2014) ). See Wells v. Kendall ,
No. 2:17-cv-2709 AC P, 2019 WL 1787172, *5 (E.D. Cal. Apr. 24, 2019) ; Powe v.
Nevada , No. 2:17-cv-00470-JAD-GWF, 2019 WL 918982, at *3 (D. Nev. Feb. 22,
2019) ("Although
the use of the 'Doe' placeholder to identify a defendant is not favored,
flexibility is allowed in some cases where the identity of the parties will not
be known prior to filing a complaint but can subsequently be determined through
discovery."). Also, again: 18 U.S.C. § 1961(1)

Output: {{'cases': {{'Gonzalez v. City of Anaheim , 747 F.3d 789,
795 (9th Cir. 2014)': 1, 'Wells v. Kendall , No. 2:17-cv-2709 AC P, 2019 WL
1787172 (E.D. Cal. Apr. 24, 2019)': 1, 'Powe v.
Nevada , No. 2:17-cv-00470-JAD-GWF, 2019 WL 918982 (D. Nev. Feb. 22,
2019)': 1}}, 'statutes': {{'18 U.S.C. § 1961(1)': 3}}}}

JSON Output Format: {schema}

Input Object: {current}

"""

FORMAT_PROMPT = """
Given a JSON object representing American caselaw legal citations extracted from a document and their
counts, return JSON where the keys are the volume-reporter-page number
identifiers of the cases and the values are the respective count.

(Put differently, replace the full citations with just the volume-reporter-page number identifiers.)

EXAMPLE: {{'cases': {{'N. Star Int'l v. Ariz. Corp. Comm'n, 720 F.2d 578 (9th
Cir. 1983)': 1, {{'Some Other v. Case, 725 F.Supp 800 (9th Cir. 1983)': 3}}}}

OUTPUT: {{'720 F.2d 578': 1, '725 F.Supp 800': 3}}
"""


def count_tokens(s: str) -> int:
    enc = tiktoken.encoding_for_model("gpt-4-turbo")
    token_ids = enc.encode(s)
    return len(token_ids)


def chunk_by_token(text: str, max_tokens=4000) -> List[str]:
    enc = tiktoken.encoding_for_model("gpt-4-turbo")
    token_ids = enc.encode(text)

    all_chunks = [
        token_ids[i : i + max_tokens] for i in range(0, len(token_ids), max_tokens)
    ]

    return [enc.decode(chunk) for chunk in all_chunks]


async def extract_citations_from_document(doc: str) -> CitationExtractionResult:
    chunks = chunk_by_token(doc)
    full_count: CitationExtractionResult = await extract_citations_from_chunks(chunks)

    cases = full_count.cases

    if len(cases) > 0:
        reformatted = await chat(
            system_prompt=FORMAT_PROMPT,
            messages=[{"role": "user", "content": str(cases)}],
        )
        full_count.cases = json.loads(reformatted)

    return full_count


async def _extract_citations_from_chunk(
    text: str, previous_result: Dict = dict()
) -> CitationExtractionResult:
    res = await chat(
        system_prompt=LLM_EXTRACTION_PROMPT.format(
            current=str(previous_result),
            schema=CitationExtractionResult.model_json_schema(),
        ),
        messages=[{"role": "user", "content": f"Input Text: {text}"}],
    )
    return CitationExtractionResult.model_validate_json(res)


async def extract_citations_from_chunks(chunks: List[str]) -> CitationExtractionResult:
    count = CitationExtractionResult(cases={}, statutes={})

    for c in chunks:
        count = await _extract_citations_from_chunk(c, count.dict())

    return count

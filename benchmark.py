import asyncio
from typing import Dict, Set

from pydantic import BaseModel

from src.openai import chat


class CitationExtractionFormat(BaseModel):
    cases: Dict[str, str]
    statutes: Set[str]


LLM_EXTRACTION_PROMPT = """
Extract the legal citations from the following block of text; answer in the provided
JSON format. 

Caselaw should be output as a Dict where key is the volume-reporter-page number
span, and the value is the case name. For short caselaw citations, this will be
volume-reporter, with the value being the short case name.

Statutes should be output whole. 

Additional rules/instructions:
- Ignore any citations to anything other than caselaw or statutes.
- If a case citation has multiple reporters, use the first one.
- If a caselaw citation lacks a name, use an empty string for the name. 
- Always return a value for both keys, even if the value is an empty dict or
list.
- Count rules of civil procedure as statutes.

EXAMPLE:
Text: The purpose of a motion to dismiss pursuant to Rule 12(b)(6) is to test
the legal sufficiency of the complaint. N. Star Int'l v. Ariz. Corp. Comm'n ,
720 F.2d 578, 581 (9th Cir. 1983). Also, a statute. 18 U.S.C. § 1961(1).

Output: {'cases': {'720 F.2d 578': 'N. Star Int'l v. Ariz. Corp. Comm'n'},
'statutes': ['18 U.S.C. § 1961(1)']}

EXAMPLE 2:
Text: Hi there. 26 U.S.C. § 172(b). Plaintiff contends it was improper for the
IRS to do this since this court invalidated that regulation. American Standard
v. United States, 220 Ct. Cl. 411, 602 F. 2d 256 (1979), rehearing en banc
denied (Oct. 12, 1979), and Union Carbide v. United States, 222 Ct. Cl. 75, 612
F. 2d 558 (1979). Plaintiff claims the methods indorsed by these cases, the
aggregate method with losses and the fractional method with losses, are the
proper methods for calculating the deduction and result in an increase in
Allied’s tax refund for 1971. 26 U.S.C. § 172. 

Output: {'cases': {'220 Ct. Cl. 411': 'American Standard v. United States', '222
Ct. Cl. 75': 'Union Carbide v. United States'}, 'statutes': ['26 U.S.C. §
172(b)', '26 U.S.C. § 172']}

EXAMPLE 3:
Text: it was an intergovernmental task force composed of several local, county,
and state governmental entities, rather than a separate legal entity unto
itself. 65 F.3d 784, 791-92 (9th Cir. 1995). Because this case includes claims
against a cow, my life is rendered null. Jones, 67 F.3d at 89.

Output: {'cases': {'65 F.3d 784': '', '67 F.3d': 'Jones'}, 'statutes': []}

EXAMPLE 4:
Text: (Doc. No. 12 at 11) (quoting Gonzalez v. City of Anaheim , 747 F.3d 789,
795 (9th Cir. 2014) ). See Wells v. Kendall , No. 2:17-cv-2709 AC P, 2019 WL
1787172, *5 (E.D. Cal. Apr. 24, 2019) ; Powe v. Nevada , No.
2:17-cv-00470-JAD-GWF, 2019 WL 918982, at *3 (D. Nev. Feb. 22, 2019) ("Although
the use of the 'Doe' placeholder to identify a defendant is not favored,
flexibility is allowed in some cases where the identity of the parties will not
be known prior to filing a complaint but can subsequently be determined through
discovery.").

Output: {'cases': {'747 F.3d 789': 'Gonzalez v. City of Anaheim', '2019 WL 1787172': 'Wells v. Kendall', '2019 WL 918982': 'Powe v. Nevada'}, 'statutes': []

JSON Output Format:\n
"""

LLM_EXTRACTION_PROMPT += str(CitationExtractionFormat.model_json_schema())


TEST_TEXT = """
In determining whether to grant summary judgment, the court views the facts - and any inferences from those facts - in the light most favorable to the nonmoving party. Matsushita Elec. Indus. Co., Ltd. v. Zenith Radio Corp. , 475 U.S. 574, 587, 106 S.Ct. 1348, 89 L.Ed.2d 538 (1986). The movant bears the burden of establishing that (1) it is entitled to judgment as a matter of law and (2) there are no genuine issues of material fact. Fed. R. Civ. P. 56(a) ;
"""

res = asyncio.run(
    chat(
        system_prompt=LLM_EXTRACTION_PROMPT,
        messages=[
            {"role": "user", "content": TEST_TEXT},
        ],
    )
)

print(res)

# res = client.chat.completions.create(
#     model="gpt-4-turbo-preview",
#     response_model=CitationExtractionFormat,
#     messages=[
#         {"role": "user", "content": LLM_EXTRACTION_PROMPT},
#         {"role": "user", "content": TEST_TEXT},
#     ],
# )

# print(res)

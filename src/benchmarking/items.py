from .types import CitationExtractionResult, TestItem

TEST_ITEMS: list[TestItem] = [
    (
        "In Gonzalez v. City of Anaheim, 747 F.3d 789 (9th Cir. 2014), the court ruled...",
        CitationExtractionResult(
            cases={"Gonzalez v. City of Anaheim, 747 F.3d 789 (9th Cir. 2014)": 1},
            statutes={},
        ),
    ),
    (
        "The statute 18 U.S.C. § 1961(1) defines racketeering activities...",
        CitationExtractionResult(cases={}, statutes={"18 U.S.C. § 1961(1)": 1}),
    ),
]

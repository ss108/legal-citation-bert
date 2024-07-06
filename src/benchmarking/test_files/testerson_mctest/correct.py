from src.benchmarking.types import CitationExtractionResult

correct = CitationExtractionResult.from_dict(
    {
        "cases": {
            "588 F.3d 97": 1,
            "604 F.2d 200": 2,
            "335 F.3d 141": 1,
            "954 F. Supp. 2d 145": 2,
        },
        "statutes": {
            "15 U.S.C. § 1114": 1,
            "21 U.S.C. § 881(a)(7)": 1,
            "33 U.S.C. § 1345(d)(3)": 1,
            "50 U.S.C. § 1806(g)": 3,
        },
    }
)

from src.benchmarking.types import CitationExtractionResult

correct = CitationExtractionResult.from_dict(
    {
        "cases": {
            "141 S. Ct. 1608": 1,
            "760 F.3d 585": 1,
            "992 F.2d 337": 1,
            "911 F.2d 870": 2,
            "886 F.2d 618": 1,
            "529 F.3d 980": 1,
            "2018 WL 5777025": 1,
            "270 F.3d 1331": 1,
            "43 F.3d 794": 3,
            "2019 WL 1595682": 1,
        },
        "statutes": {
            "8 U.S.C. § 1229a(b)(5)(C)(ii)": 8,
            "21 U.S.C. § 881(a)(7)": 1,
            "33 U.S.C. § 1345(d)(3)": 1,
            "50 U.S.C. § 1806(g)": 3,
        },
    }
)

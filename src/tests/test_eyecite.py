import pytest
from eyecite import get_citations

"""
"""


@pytest.mark.parametrize(
    ["sentence", "expected"],
    [
        (
            "Under title III, public accommodations are required to provide auxiliary aids in order to extend their services to persons with disabilities,",
            False,
        ),
        (
            "The term “most integrated setting appropriate,” as used in title III, is not unconstitutionally vague.",
            False,
        ),
        (
            "see also id. pt. Ill, at 56-57 (1990), reprinted in 1990 U.S.C.C.A.N. 267, 479-80 (“House report, Pt. III”).",
            False,
        ),
        (
            "H.R.Rep. No. 485, 101st Cong., 2d Sess., pt. II, at 102 (1990), reprinted in 1990 U.S.C.C.A.N. 267, 385 (“House Report, Pt. II”); 5. Undue Burden",
            False,
        ),
        (
            "However, as the Government notes, the Court more recently addressed the issue of retroactivity in Federal Housing Administration v. Darlington, Inc., 358 U.S. 84, 79 S.Ct. 141, 3 L.Ed.2d 132 (1958), reh’g denied, 358 U.S. 937, 79 S.Ct. 310, 3 L.Ed.2d 311 (1959). In that case, the Court rejected a due process challenge to a statute which directed the Federal Housing Administration (“FHA”) to insure mortgages only for strictly residential housing, rather than transient housing which had been previously included. ",
            True,
        ),
    ],
)
def test_false(sentence: str, expected: bool):
    cits = get_citations(sentence)

    if expected:
        assert len(cits) > 0
    else:
        assert len(cits) == 0

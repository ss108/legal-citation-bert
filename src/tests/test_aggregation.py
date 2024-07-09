from typing import List

import pytest

from src.benchmarking.temp_aggregation import (
    CaselawCitation,
    LabelPrediction,
    aggregate_entities,
)


@pytest.mark.parametrize(
    ["labels", "expected"],
    [
        (
            [
                LabelPrediction(token="h", label="B-CASE_NAME"),
                LabelPrediction(token="e", label="I-CASE_NAME"),
                LabelPrediction(token="h", label="I-CASE_NAME"),
            ],
            [LabelPrediction(token="heh", label="CASE_NAME")],
        ),
        (
            [
                LabelPrediction(token="h", label="B-CASE_NAME"),
                LabelPrediction(token="e", label="I-CASE_NAME"),
                LabelPrediction(token="h", label="I-CASE_NAME"),
                LabelPrediction(token=",", label="O"),
            ],
            [LabelPrediction(token="heh", label="CASE_NAME")],
        ),
        (
            [
                LabelPrediction(token="h", label="B-CASE_NAME"),
                LabelPrediction(token="e", label="I-CASE_NAME"),
                LabelPrediction(token="h", label="I-CASE_NAME"),
                LabelPrediction(token=",", label="O"),
                LabelPrediction(token="87", label="B-VOLUME"),
                LabelPrediction(token="F", label="B-REPORTER"),
            ],
            [
                LabelPrediction(token="heh", label="CASE_NAME"),
                LabelPrediction(token="87", label="VOLUME"),
                LabelPrediction(token="F", label="REPORTER"),
            ],
        ),
        (
            [
                LabelPrediction(token="h", label="B-CASE_NAME"),
                LabelPrediction(token="e", label="I-CASE_NAME"),
                LabelPrediction(token="h", label="I-CASE_NAME"),
                LabelPrediction(token=",", label="O"),
                LabelPrediction(token="87", label="B-VOLUME"),
                LabelPrediction(token="F", label="B-REPORTER"),
                LabelPrediction(token=".", label="I-REPORTER"),
                LabelPrediction(token="3", label="I-REPORTER"),
                LabelPrediction(token="##d", label="I-REPORTER"),
            ],
            [
                LabelPrediction(token="heh", label="CASE_NAME"),
                LabelPrediction(token="87", label="VOLUME"),
                LabelPrediction(token="F.3d", label="REPORTER"),
            ],
        ),
        (
            [
                LabelPrediction(token="h", label="B-CASE_NAME"),
                LabelPrediction(token="e", label="I-CASE_NAME"),
                LabelPrediction(token="h", label="I-CASE_NAME"),
                LabelPrediction(token=",", label="O"),
                LabelPrediction(token="87", label="B-VOLUME"),
                LabelPrediction(token="F", label="B-REPORTER"),
                LabelPrediction(token=".", label="I-REPORTER"),
                LabelPrediction(token="3", label="I-REPORTER"),
                LabelPrediction(token="##d", label="I-REPORTER"),
                LabelPrediction(token="at", label="O"),
                LabelPrediction(token="99", label="B-PIN"),
            ],
            [
                LabelPrediction(token="heh", label="CASE_NAME"),
                LabelPrediction(token="87", label="VOLUME"),
                LabelPrediction(token="F.3d", label="REPORTER"),
                LabelPrediction(token="99", label="PIN"),
            ],
        ),
    ],
)
def test_agg_short_cite(labels: List[LabelPrediction], expected: List[LabelPrediction]):
    res = aggregate_entities(labels)
    assert res == expected


@pytest.mark.parametrize(
    ["labels", "expected"],
    [
        (
            [
                LabelPrediction(token="Lee", label="B-CASE_NAME"),
                LabelPrediction(token="##gin", label="I-CASE_NAME"),
            ],
            [LabelPrediction(token="Leegin", label="CASE_NAME")],
        ),
        (
            [
                LabelPrediction(token="Lee", label="B-CASE_NAME"),
                LabelPrediction(token="##gin", label="I-CASE_NAME"),
                LabelPrediction(token="Creative", label="I-CASE_NAME"),
                LabelPrediction(token="Lea", label="I-CASE_NAME"),
                LabelPrediction(token="##ther", label="I-CASE_NAME"),
                LabelPrediction(token="Products", label="I-CASE_NAME"),
            ],
            [
                LabelPrediction(
                    token="Leegin Creative Leather Products", label="CASE_NAME"
                )
            ],
        ),
        (
            [
                LabelPrediction(token="Lee", label="B-CASE_NAME"),
                LabelPrediction(token="##gin", label="I-CASE_NAME"),
                LabelPrediction(token="Creative", label="I-CASE_NAME"),
                LabelPrediction(token="Lea", label="I-CASE_NAME"),
                LabelPrediction(token="##ther", label="I-CASE_NAME"),
                LabelPrediction(token="Products", label="I-CASE_NAME"),
                LabelPrediction(token=",", label="I-CASE_NAME"),
            ],
            [
                LabelPrediction(
                    token="Leegin Creative Leather Products,", label="CASE_NAME"
                )
            ],
        ),
        # (
        #     [
        #         LabelPrediction(token="Lee", label="B-CASE_NAME"),
        #         LabelPrediction(token="##gin", label="I-CASE_NAME"),
        #         LabelPrediction(token="Creative", label="I-CASE_NAME"),
        #         LabelPrediction(token="Lea", label="I-CASE_NAME"),
        #         LabelPrediction(token="##ther", label="I-CASE_NAME"),
        #         LabelPrediction(token="Products", label="I-CASE_NAME"),
        #         LabelPrediction(token=",", label="I-CASE_NAME"),
        #         LabelPrediction(token="Inc", label="I-CASE_NAME"),
        #         LabelPrediction(token=".", label="I-CASE_NAME"),
        #         LabelPrediction(token="v", label="I-CASE_NAME"),
        #         LabelPrediction(token=".", label="I-CASE_NAME"),
        #         LabelPrediction(token="Some", label="I-CASE_NAME"),
        #         LabelPrediction(token="Guy", label="I-CASE_NAME"),
        #     ],
        #     [
        #         LabelPrediction(
        #             token="Leegin Creative Leather Products, Inc. v. Some Guy",
        #             label="CASE_NAME",
        #         )
        #     ],
        # ),
        (
            [
                LabelPrediction(token="10", label="B-PIN"),
                LabelPrediction(token="##1", label="I-PIN"),
                LabelPrediction(token="-", label="I-PIN"),
                LabelPrediction(token="10", label="I-PIN"),
                LabelPrediction(token="##8", label="I-PIN"),
            ],
            [LabelPrediction(token="101-108", label="PIN")],
        ),
    ],
)
def test_agg_full_case(labels: List[LabelPrediction], expected: List[LabelPrediction]):
    res = aggregate_entities(labels)
    assert res == expected


@pytest.mark.parametrize(
    ["labels", "expected"],
    [
        (
            [
                LabelPrediction(token="Foo v. Bar", label="CASE_NAME"),
                LabelPrediction(token="551", label="VOLUME"),
                LabelPrediction(token="U.S.", label="REPORTER"),
            ],
            "551 U.S.",
        ),
        (
            [
                LabelPrediction(token="Foo v. Bar", label="CASE_NAME"),
                LabelPrediction(token="551", label="VOLUME"),
                LabelPrediction(token="U.S.", label="REPORTER"),
                LabelPrediction(token="877", label="PAGE"),
                LabelPrediction(token="904-907", label="PIN"),
            ],
            "551 U.S. 877",
        ),
        (
            [
                LabelPrediction(token="Angel v. Eva-001", label="CASE_NAME"),
                LabelPrediction(token="12", label="VOLUME"),
                LabelPrediction(token="F.Supp.", label="REPORTER"),
                LabelPrediction(token="10", label="PAGE"),
                LabelPrediction(token="19", label="PIN"),
            ],
            "12 F.Supp. 10",
        ),
        (
            [
                LabelPrediction(token="garbage", label="O"),
                LabelPrediction(token="Angel v. Eva-002", label="CASE_NAME"),
                LabelPrediction(token="12", label="VOLUME"),
                LabelPrediction(token="F.Supp.", label="REPORTER"),
                LabelPrediction(token="10", label="PAGE"),
                LabelPrediction(token="19", label="PIN"),
                LabelPrediction(token="19BLAHFH", label="O"),
            ],
            "12 F.Supp. 10",
        ),
    ],
)
def test_full_cite_guid(labels: List[LabelPrediction], expected: str):
    cit = CaselawCitation.from_token_label_pairs(labels)

    assert cit, "Citation should not be None"
    assert cit.guid == expected


@pytest.mark.parametrize(
    ["labels", "guid"],
    [
        (
            [
                LabelPrediction(token="[CLS]", label="O"),
                LabelPrediction(token="Virgin", label="B-CASE_NAME"),
                LabelPrediction(token="Enterprises", label="I-CASE_NAME"),
                LabelPrediction(token="Ltd", label="I-CASE_NAME"),
                LabelPrediction(token=".", label="I-CASE_NAME"),
                LabelPrediction(token="v", label="I-CASE_NAME"),
                LabelPrediction(token=".", label="I-CASE_NAME"),
                LabelPrediction(token="Na", label="I-CASE_NAME"),
                LabelPrediction(token="##wab", label="I-CASE_NAME"),
                LabelPrediction(token=",", label="O"),
                LabelPrediction(token="335", label="B-VOLUME"),
                LabelPrediction(token="F", label="B-REPORTER"),
                LabelPrediction(token=".", label="I-REPORTER"),
                LabelPrediction(token="3", label="I-REPORTER"),
                LabelPrediction(token="##d", label="I-REPORTER"),
                LabelPrediction(token="141", label="B-PAGE"),
                LabelPrediction(token=",", label="O"),
                LabelPrediction(token="147", label="B-PIN"),
                LabelPrediction(token="(", label="O"),
                LabelPrediction(token="2d", label="B-COURT"),
                LabelPrediction(token="C", label="I-COURT"),
                LabelPrediction(token="##ir", label="I-COURT"),
                LabelPrediction(token=".", label="I-COURT"),
                LabelPrediction(token="2003", label="B-YEAR"),
                LabelPrediction(token=")", label="O"),
                LabelPrediction(token=".", label="O"),
                LabelPrediction(token="[SEP]", label="I-CASE_NAME"),
            ],
            "335 F.3d 141",
        ),
        (
            [
                LabelPrediction(token="[CLS]", label="O"),
                LabelPrediction(token='"', label="O"),
                LabelPrediction(token="Con", label="O"),
                LabelPrediction(token="##fusion", label="O"),
                LabelPrediction(token='"', label="O"),
                LabelPrediction(token="in", label="O"),
                LabelPrediction(token="this", label="O"),
                LabelPrediction(token="context", label="O"),
                LabelPrediction(token="is", label="O"),
                LabelPrediction(token="not", label="O"),
                LabelPrediction(token="limited", label="O"),
                LabelPrediction(token="to", label="O"),
                LabelPrediction(token="a", label="O"),
                LabelPrediction(token="mistaken", label="O"),
                LabelPrediction(token="belief", label="O"),
                LabelPrediction(token="among", label="O"),
                LabelPrediction(token="consumers", label="O"),
                LabelPrediction(token="that", label="O"),
                LabelPrediction(token="the", label="O"),
                LabelPrediction(token="plaintiff", label="O"),
                LabelPrediction(token="is", label="O"),
                LabelPrediction(token="the", label="O"),
                LabelPrediction(token="producer", label="O"),
                LabelPrediction(token="of", label="O"),
                LabelPrediction(token="the", label="O"),
                LabelPrediction(token="defendant", label="O"),
                LabelPrediction(token="", label="O"),
                LabelPrediction(token="s", label="O"),
                LabelPrediction(token="goods", label="O"),
                LabelPrediction(token=".", label="O"),
                LabelPrediction(token="Star", label="B-CASE_NAME"),
                LabelPrediction(token="##bu", label="I-CASE_NAME"),
                LabelPrediction(token="##cks", label="I-CASE_NAME"),
                LabelPrediction(token="Corp", label="I-CASE_NAME"),
                LabelPrediction(token=".", label="I-CASE_NAME"),
                LabelPrediction(token="v", label="I-CASE_NAME"),
                LabelPrediction(token=".", label="I-CASE_NAME"),
                LabelPrediction(token="Wolfe", label="I-CASE_NAME"),
                LabelPrediction(
                    token="'",
                    label="I-CASE_NAME",
                ),
                LabelPrediction(token="s", label="I-CASE_NAME"),
                LabelPrediction(token="Borough", label="I-CASE_NAME"),
                LabelPrediction(token="Coffee", label="I-CASE_NAME"),
                LabelPrediction(token=",", label="I-CASE_NAME"),
                LabelPrediction(token="Inc", label="I-CASE_NAME"),
                LabelPrediction(token=".", label="I-CASE_NAME"),
                LabelPrediction(token=",", label="O"),
                LabelPrediction(token="58", label="B-VOLUME"),
                LabelPrediction(token="##8", label="I-VOLUME"),
                LabelPrediction(token="F", label="B-REPORTER"),
                LabelPrediction(token=".", label="I-REPORTER"),
                LabelPrediction(token="3", label="I-REPORTER"),
                LabelPrediction(token="##d", label="I-REPORTER"),
                LabelPrediction(token="97", label="B-PAGE"),
                LabelPrediction(token=",", label="O"),
                LabelPrediction(token="114", label="B-PIN"),
                LabelPrediction(token="(", label="O"),
                LabelPrediction(token="2d", label="B-COURT"),
                LabelPrediction(token="C", label="I-COURT"),
                LabelPrediction(token="##ir", label="I-COURT"),
                LabelPrediction(token=".", label="I-COURT"),
                LabelPrediction(token="2009", label="B-YEAR"),
                LabelPrediction(token=")", label="O"),
                LabelPrediction(token=".", label="O"),
                LabelPrediction(token="[SEP]", label="O"),
            ],
            "588 F.3d 97",
        ),
    ],
)
def test_loop(labels: List[LabelPrediction], guid: str):
    ents = aggregate_entities(labels)
    cit = CaselawCitation.from_token_label_pairs(ents)
    assert cit

    assert cit.guid == guid


# def test_loop_null():

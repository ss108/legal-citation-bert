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
    ],
)
def test_full_cite_guid(labels: List[LabelPrediction], expected: str):
    cit = CaselawCitation.from_token_label_pairs(labels)

    assert cit, "Citation should not be None"
    assert cit.guid == expected

from typing import List

import pytest

from src.benchmarking.temp_aggregation import LabelPrediction, aggregate_entities


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
    ],
)
def test_agg_short_cite(labels: List[LabelPrediction], expected: List[LabelPrediction]):
    res = aggregate_entities(labels)
    assert res == expected

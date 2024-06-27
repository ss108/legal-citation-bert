from typing import List, Tuple

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
            [LabelPrediction(token="heh", label="B-CASE_NAME")],
        ),
    ],
)
def test_agg(labels: List[LabelPrediction], expected: List[LabelPrediction]):
    res = aggregate_entities(labels)
    assert res == expected

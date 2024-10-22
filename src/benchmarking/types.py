from __future__ import annotations

from typing import Dict, List, TypeAlias

from pydantic import BaseModel
from wasabi import msg


class CitationExtractionResult(BaseModel):
    cases: Dict[str, int]
    statutes: Dict[str, int]

    @classmethod
    def from_dict(cls, d: Dict) -> CitationExtractionResult | None:
        try:
            return cls(cases=d["cases"], statutes=d["statutes"])
        except Exception as e:
            msg.fail(
                f"Error instantiating CitationExtractionResult from LLM response: {e}"
            )
            return None

    def sort(self):
        self.cases = dict(sorted(self.cases.items(), key=lambda x: x[0], reverse=True))
        self.statutes = dict(
            sorted(self.statutes.items(), key=lambda x: x[0], reverse=True)
        )
        return self

    def err_count(self, model_response: CitationExtractionResult) -> int:
        err_count = 0

        for k, v in self.cases.items():
            if k not in model_response.cases:
                err_count += v
            else:
                correct_count = v
                response_count = model_response.cases[k]
                err_count += abs(correct_count - response_count)

        for k, v in self.statutes.items():
            if k not in model_response.statutes:
                err_count += v
            else:
                correct_count = v
                response_count = model_response.statutes[k]
                err_count += abs(correct_count - response_count)

        return err_count

    @staticmethod
    def combine(results: List[CitationExtractionResult]) -> CitationExtractionResult:
        combined_cases: Dict[str, int] = {}
        combined_statutes: Dict[str, int] = {}

        for result in results:
            # Aggregate cases
            for case, count in result.cases.items():
                if case in combined_cases:
                    combined_cases[case] += count
                else:
                    combined_cases[case] = count

            # Aggregate statutes
            for statute, count in result.statutes.items():
                if statute in combined_statutes:
                    combined_statutes[statute] += count
                else:
                    combined_statutes[statute] = count

        return CitationExtractionResult(
            cases=combined_cases, statutes=combined_statutes
        )


TestItem: TypeAlias = tuple[str, CitationExtractionResult]


class DocResult(BaseModel):
    text_preview: str
    err_count: int
    correct_count: int
    accuracy: float


class BenchmarkResult:
    def __init__(self, thingy: str):
        self._total_errors = 0
        self._total_tests = 0
        self.individual_results: List[DocResult] = []
        self.thingy = thingy

    @property
    def total_errors(self) -> int:
        return self._total_errors

    @property
    def total_tests(self) -> int:
        return self._total_tests

    def add_result(self, text: str, correct: int, errors: int):
        self._total_errors += errors
        self._total_tests += correct + errors
        accuracy = 100 * correct / (correct + errors) if (correct + errors) > 0 else 0
        self.individual_results.append(
            DocResult(
                text_preview=text[:25],
                err_count=errors,
                correct_count=correct,
                accuracy=accuracy,
            )
        )

    @property
    def overall_accuracy(self) -> float:
        if self.total_tests == 0:
            return 0.0
        correct_count = self.total_tests - self.total_errors
        return 100 * correct_count / self.total_tests

    def log_individual_results(self):
        for result in self.individual_results:
            accuracy = result.accuracy
            if accuracy == 100:
                msg.good(
                    f"{self.thingy} perfect accuracy: {accuracy:.2f}% for text: {result.text_preview}..."
                )
            else:
                msg.warn(
                    f"{self.thingy} accuracy: {accuracy:.2f}% for text: {result.text_preview}..."
                )
                msg.info(
                    f"{self.thingy} errors: {result.err_count} | Correct: {result.correct_count}"
                )

    def log_overall_results(self):
        message_text = f"Overall accuracy for {self.thingy} across all test items: {self.overall_accuracy:.2f}%"
        msg.good(
            title=f"{self.thingy} Benchmarking Results",
            text=message_text,
        )

    @staticmethod
    def combine(results: List[BenchmarkResult]) -> BenchmarkResult:
        if not results:
            raise ValueError("No results to combine")

        combined_result = BenchmarkResult(results[0].thingy)

        for result in results:
            combined_result._total_errors += result.total_errors
            combined_result._total_tests += result.total_tests
            combined_result.individual_results.extend(result.individual_results)

        return combined_result

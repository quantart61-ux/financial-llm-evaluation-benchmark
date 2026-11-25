import json
from pathlib import Path

import pytest

from src.schema import CandidateSubmission
from src.verifier import FinancialEvalVerifier, fdx_fy2025_ground_truth


DIMENSION_SCORES = {
    "numerical_grounding": 5.0,
    "accounting_rigor": 5.0,
    "edge_case_discipline": 5.0,
    "instruction_adherence": 5.0,
    "audit_transparency": 5.0,
}


def submission(metric_value: float) -> CandidateSubmission:
    return CandidateSubmission(
        candidate_id="test",
        metrics={"value": metric_value},
        dimension_scores=DIMENSION_SCORES.copy(),
    )


@pytest.mark.parametrize("value", [100.19, 100.20])
def test_metric_passes_at_or_inside_relative_tolerance(value: float) -> None:
    report = FinancialEvalVerifier({"value": 100.0}).verify(submission(value))

    assert report.checks_passed == 1
    assert report.verdict == "ACCEPT"


def test_metric_fails_outside_relative_tolerance() -> None:
    report = FinancialEvalVerifier({"value": 100.0}).verify(submission(100.25))

    assert report.checks_passed == 0
    assert report.verdict == "REJECT"


@pytest.mark.parametrize("tolerance", [-0.001, float("nan"), float("inf")])
def test_invalid_tolerance_raises_value_error(tolerance: float) -> None:
    with pytest.raises(ValueError, match="rel_tol"):
        FinancialEvalVerifier({"value": 100.0}, rel_tol=tolerance)


def test_weighted_score_uses_five_rubric_weights() -> None:
    scores = {
        "numerical_grounding": 4.0,
        "accounting_rigor": 3.0,
        "edge_case_discipline": 2.0,
        "instruction_adherence": 1.0,
        "audit_transparency": 0.0,
    }
    report = FinancialEvalVerifier({"value": 100.0}).verify(
        CandidateSubmission("scored", {"value": 100.0}, scores)
    )

    assert report.weighted_score == pytest.approx(2.65)
    assert report.verdict == "REJECT"


def test_fdx_fy2025_ground_truth_is_preserved() -> None:
    assert fdx_fy2025_ground_truth() == {
        "GAAP_EBITDA": 9845.00,
        "Adjusted_EBITDA": 10541.00,
        "Net_Leverage": 1.31,
        "Free_Cash_Flow": 3156.00,
        "Diluted_Shares": 250.96,
    }


def test_all_benchmark_cases_reject_traps_and_accept_aligned_candidates() -> None:
    cases_path = Path(__file__).parents[1] / "data" / "benchmark_cases.json"
    cases = json.loads(cases_path.read_text(encoding="utf-8"))

    assert [case["case_id"] for case in cases] == [
        f"TRAP-{number:02d}" for number in range(1, 11)
    ]
    for case in cases:
        verifier = FinancialEvalVerifier(case["ground_truth"])
        aligned_count = 0
        trap_count = 0
        for candidate_data in case["candidates"]:
            report = verifier.verify(CandidateSubmission(**candidate_data))
            if candidate_data["candidate_id"] == "Model_C_Aligned":
                aligned_count += 1
                assert report.verdict == "ACCEPT", case["case_id"]
                assert report.weighted_score == pytest.approx(5.0)
            else:
                trap_count += 1
                assert report.verdict == "REJECT", case["case_id"]
        assert aligned_count == 1
        assert trap_count >= 1

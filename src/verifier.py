import math
from numbers import Real

from src.schema import AuditReport, CandidateSubmission, RubricWeights


def fdx_fy2025_ground_truth() -> dict[str, float]:
    filing = {
        "revenue": 87928.0,
        "operating_income": 5559.0,
        "depreciation_amortization": 4286.0,
        "restructuring": 584.0,
        "litigation_settlement": 112.0,
        "stock_based_compensation": 219.0,
        "interest_expense": 725.0,
        "cash_taxes": 1185.0,
        "capex": 5178.0,
        "nwc_beginning": 2150.0,
        "nwc_ending": 2476.0,
        "total_debt": 20345.0,
        "unrestricted_cash": 6514.0,
        "basic_shares": 248.40,
        "current_share_price": 264.50,
        "tranche_a_options": 8.20,
        "tranche_a_strike": 182.00,
        "tranche_b_options": 5.50,
        "tranche_b_strike": 295.00,
    }
    gaap_ebitda = filing["operating_income"] + filing["depreciation_amortization"]
    adjusted_ebitda = gaap_ebitda + filing["restructuring"] + filing["litigation_settlement"]
    net_leverage = (
        filing["total_debt"] - filing["unrestricted_cash"]
    ) / adjusted_ebitda
    free_cash_flow = (
        gaap_ebitda
        - filing["cash_taxes"]
        - filing["capex"]
        - (filing["nwc_ending"] - filing["nwc_beginning"])
    )
    in_the_money_proceeds = filing["tranche_a_options"] * filing["tranche_a_strike"]
    diluted_shares = filing["basic_shares"] + filing["tranche_a_options"] - (
        in_the_money_proceeds / filing["current_share_price"]
    )

    return {
        "GAAP_EBITDA": round(gaap_ebitda, 2),
        "Adjusted_EBITDA": round(adjusted_ebitda, 2),
        "Net_Leverage": round(net_leverage, 2),
        "Free_Cash_Flow": round(free_cash_flow, 2),
        "Diluted_Shares": round(diluted_shares, 2),
    }


class FinancialEvalVerifier:
    DIMENSIONS = (
        "numerical_grounding",
        "accounting_rigor",
        "edge_case_discipline",
        "instruction_adherence",
        "audit_transparency",
    )

    def __init__(
        self,
        ground_truth: dict[str, float],
        weights: RubricWeights | None = None,
        rel_tol: float = 0.002,
    ) -> None:
        if (
            not isinstance(rel_tol, Real)
            or isinstance(rel_tol, bool)
            or not math.isfinite(rel_tol)
            or rel_tol < 0
        ):
            raise ValueError("rel_tol must be a finite, non-negative number")
        if not ground_truth:
            raise ValueError("ground_truth must contain at least one metric")
        if any(
            not isinstance(value, Real)
            or isinstance(value, bool)
            or not math.isfinite(value)
            for value in ground_truth.values()
        ):
            raise ValueError("ground_truth values must be finite numbers")

        self.ground_truth = ground_truth
        self.weights = weights or RubricWeights()
        if not isinstance(self.weights, RubricWeights):
            raise ValueError("weights must be a RubricWeights instance")
        weight_values = self.weights.as_mapping()
        if any(
            not isinstance(value, Real)
            or isinstance(value, bool)
            or not math.isfinite(value)
            or value < 0
            for value in weight_values.values()
        ):
            raise ValueError("rubric weights must be finite, non-negative numbers")
        if not math.isclose(sum(weight_values.values()), 1.0, abs_tol=1e-9):
            raise ValueError("rubric weights must sum to 1.0")
        self.rel_tol = float(rel_tol)

    def verify(self, submission: CandidateSubmission) -> AuditReport:
        score_dimensions = submission.dimension_scores
        if set(score_dimensions) != set(self.DIMENSIONS):
            raise ValueError(f"dimension_scores must contain exactly {self.DIMENSIONS}")
        if any(
            not isinstance(score, Real)
            or isinstance(score, bool)
            or not math.isfinite(score)
            or not 0 <= score <= 5
            for score in score_dimensions.values()
        ):
            raise ValueError("dimension scores must be finite numbers between 0 and 5")

        failures: list[str] = []
        checks_passed = 0
        for metric, expected in self.ground_truth.items():
            actual = submission.metrics.get(metric)
            if (
                not isinstance(actual, Real)
                or isinstance(actual, bool)
                or not math.isfinite(actual)
            ):
                failures.append(f"{metric}: missing or non-finite value")
                continue
            relative_error = (
                abs(actual - expected) / abs(expected)
                if expected != 0
                else (0.0 if actual == 0 else math.inf)
            )
            if relative_error <= self.rel_tol or math.isclose(
                relative_error, self.rel_tol, rel_tol=1e-12, abs_tol=1e-15
            ):
                checks_passed += 1
            else:
                failures.append(f"{metric}: got {actual}, expected {expected}")

        weight_mapping = self.weights.as_mapping()
        weighted_score = sum(
            score_dimensions[dimension] * weight_mapping[dimension]
            for dimension in self.DIMENSIONS
        )
        verdict = (
            "ACCEPT"
            if checks_passed == len(self.ground_truth) and weighted_score >= 4.50
            else "REJECT"
        )
        return AuditReport(
            candidate_id=submission.candidate_id,
            checks_passed=checks_passed,
            total_checks=len(self.ground_truth),
            weighted_score=weighted_score,
            verdict=verdict,
            failures=failures,
        )

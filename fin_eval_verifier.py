Script này tự động tính toán bộ chỉ số chuẩn từ dữ liệu báo cáo tài chính 10-K (bao gồm điều chỉnh Adjusted EBITDA, tỷ lệ đòn bẩy nợ ròng, dòng tiền tự do FCF và số lượng cổ phiếu pha loãng theo phương pháp cổ phiếu quỹ ASC 260), sau đó đối chiếu và chấm điểm tự động đầu ra của các mô hình LLM theo thang điểm RLHF 5 chiều.

Python
"""
Automated Financial LLM Fact-Checker & RLHF Rubric Scoring Engine
Author: Huynh Vi
Framework: US GAAP / ASC 260 Financial Reasoning Verification & RLHF Evaluation
"""

import numpy as np
import pandas as pd


def run_financial_llm_benchmark() -> None:
    # ==========================================================================
    # 1. GROUND-TRUTH FINANCIAL FILING EXCERPT (FY2025 10-K, $ MILLIONS)
    # ==========================================================================
    filing_data = {
        "company": "Vanguard Logistics & Freight Corp. (NYSE: VLF)",
        "revenue": 2450.0,
        "operating_income_gaap": 310.0,
        "depreciation_amortization": 85.0,
        "one_time_restructuring": 22.0,     # Non-recurring facility closure & severance
        "litigation_settlement": 13.0,      # Non-recurring legal settlement charge
        "stock_based_compensation": 18.0,   # Non-cash SBC (excluded per prompt constraint)
        "interest_expense": 42.0,
        "cash_taxes": 58.0,
        "capex": 95.0,
        "nwc_beginning": 180.0,
        "nwc_ending": 212.0,                # +$32.0M NWC expansion -> -$32.0M cash outflow
        "total_debt": 1150.0,
        "unrestricted_cash": 190.0,
        # Equity & Stock Option Tranches (Millions, except strike/share prices)
        "basic_shares": 100.0,
        "current_share_price": 40.00,
        "tranche_a_options": 6.0,           # Strike = $25.00 (< $40.00 -> In-the-money / Dilutive)
        "tranche_a_strike": 25.00,
        "tranche_b_options": 4.0,           # Strike = $50.00 (> $40.00 -> Out-of-the-money / Anti-dilutive)
        "tranche_b_strike": 50.00,
    }

    # ==========================================================================
    # 2. DETERMINISTIC GOLD-STANDARD CALCULATION ENGINE
    # ==========================================================================
    # A. GAAP EBITDA & Cash-Normalized Adjusted EBITDA
    gaap_ebitda = filing_data["operating_income_gaap"] + filing_data["depreciation_amortization"]
    adj_ebitda = (
        gaap_ebitda
        + filing_data["one_time_restructuring"]
        + filing_data["litigation_settlement"]
    )

    # B. Net Leverage Ratio (Net Debt / Adjusted EBITDA)
    net_debt = filing_data["total_debt"] - filing_data["unrestricted_cash"]
    net_leverage = net_debt / adj_ebitda

    # C. Free Cash Flow (GAAP EBITDA - Cash Taxes - CapEx - Change in NWC)
    delta_nwc = filing_data["nwc_ending"] - filing_data["nwc_beginning"]
    free_cash_flow = gaap_ebitda - filing_data["cash_taxes"] - filing_data["capex"] - delta_nwc

    # D. Diluted Share Count (ASC 260 Treasury Stock Method)
    # Guardrail: Out-of-the-money options (Tranche B @ $50 > $40) MUST be excluded to prevent anti-dilution.
    in_the_money_proceeds = filing_data["tranche_a_options"] * filing_data["tranche_a_strike"]
    repurchased_shares = in_the_money_proceeds / filing_data["current_share_price"]
    net_dilutive_shares = filing_data["tranche_a_options"] - repurchased_shares
    total_diluted_shares = filing_data["basic_shares"] + net_dilutive_shares

    gold_standard = {
        "GAAP_EBITDA": round(gaap_ebitda, 2),
        "Adjusted_EBITDA": round(adj_ebitda, 2),
        "Net_Leverage": round(net_leverage, 2),
        "Free_Cash_Flow": round(free_cash_flow, 2),
        "Diluted_Shares": round(total_diluted_shares, 2),
    }

    print("=" * 92)
    print(f"FINANCIAL LLM REASONING BENCHMARK: {filing_data['company']}")
    print("=" * 92)
    print("--- [STEP 1] DETERMINISTIC GOLD-STANDARD GROUND TRUTH ---")
    for metric, val in gold_standard.items():
        print(f"  {metric:<20}: {val}")

    # ==========================================================================
    # 3. CANDIDATE LLM OUTPUTS & AUTOMATED FACT-CHECKING
    # ==========================================================================
    # - LLM_Baseline_v1: Misses non-recurring adjustments, flips NWC sign (+32), sums all options naively.
    # - LLM_Sycophant_v2: Over-adjusts EBITDA by adding SBC ($18M) and wrongly nets out-of-the-money options.
    # - LLM_FinAligned_v3: Adheres strictly to US GAAP / ASC 260 rules and prompt constraints.
    llm_responses = {
        "LLM_Baseline_v1": {
            "GAAP_EBITDA": 395.00,
            "Adjusted_EBITDA": 395.00,
            "Net_Leverage": 2.43,
            "Free_Cash_Flow": 274.00,
            "Diluted_Shares": 110.00,
            "cited_gaap_standard": False,
        },
        "LLM_Sycophant_v2": {
            "GAAP_EBITDA": 395.00,
            "Adjusted_EBITDA": 448.00,
            "Net_Leverage": 2.14,
            "Free_Cash_Flow": 210.00,
            "Diluted_Shares": 101.25,
            "cited_gaap_standard": True,
        },
        "LLM_FinAligned_v3": {
            "GAAP_EBITDA": 395.00,
            "Adjusted_EBITDA": 430.00,
            "Net_Leverage": 2.23,
            "Free_Cash_Flow": 210.00,
            "Diluted_Shares": 102.25,
            "cited_gaap_standard": True,
        },
    }

    # Institutional Audit Precision Tolerance: 0.20% relative error
    audit_tolerance = 0.002
    scorecard_rows = []

    for model_id, preds in llm_responses.items():
        failure_modes = []
        passed_count = 0
        total_metrics = len(gold_standard)

        for metric, truth in gold_standard.items():
            pred = preds[metric]
            rel_error = abs(pred - truth) / abs(truth)
            if rel_error <= audit_tolerance:
                passed_count += 1
            else:
                failure_modes.append(f"{metric} (Got {pred}, Exp {truth})")

        # 5-Dimension Weighted RLHF Evaluation Rubric (0.00 to 5.00 Scale)
        # 1. Arithmetic & Numerical Grounding (35%)
        dim_numerical = round((passed_count / total_metrics) * 5.0, 2)
        # 2. Accounting Standard & Valuation Methodology (25%)
        dim_methodology = (
            5.0 if (passed_count == total_metrics and preds["cited_gaap_standard"])
            else (3.0 if preds["cited_gaap_standard"] else 1.5)
        )
        # 3. Sign Convention & Edge-Case Rigor (20%)
        fcf_ok = abs(preds["Free_Cash_Flow"] - gold_standard["Free_Cash_Flow"]) < 0.05
        eps_ok = abs(preds["Diluted_Shares"] - gold_standard["Diluted_Shares"]) < 0.05
        dim_edge_cases = 5.0 if (fcf_ok and eps_ok) else 2.0
        # 4. Constraint & Instruction Adherence (10%)
        dim_instruction = 5.0 if abs(preds["Adjusted_EBITDA"] - gold_standard["Adjusted_EBITDA"]) < 0.05 else 2.5
        # 5. Anti-Hallucination & Grounded Justification (10%)
        dim_grounding = 5.0 if not failure_modes else 2.0

        composite_rlhf_score = (
            0.35 * dim_numerical
            + 0.25 * dim_methodology
            + 0.20 * dim_edge_cases
            + 0.10 * dim_instruction
            + 0.10 * dim_grounding
        )

        if composite_rlhf_score >= 4.50:
            verdict = "ACCEPT (Gold RLHF Sample)"
        elif composite_rlhf_score >= 3.00:
            verdict = "REVISE (Partial Logic)"
        else:
            verdict = "REJECT (Critical Math/GAAP Errors)"

        scorecard_rows.append({
            "Candidate_Model": model_id,
            "Checks_Passed": f"{passed_count}/{total_metrics}",
            "RLHF_Score": f"{composite_rlhf_score:.2f} / 5.00",
            "RLHF_Verdict": verdict,
            "Detected_Failure_Modes": "; ".join(failure_modes) if failure_modes else "None (100% Grounded)",
        })

    audit_df = pd.DataFrame(scorecard_rows)
    print("\n--- [STEP 2] AUTOMATED RLHF SCORECARD & FAILURE DIAGNOSTICS ---")
    print(audit_df.to_string(index=False))
    print("=" * 92)


if __name__ == "__main__":
    run_financial_llm_benchmark()

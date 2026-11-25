# Financial LLM Reasoning Benchmark & RLHF Evaluation Rubric

A domain-specific evaluation suite and deterministic Python verification harness for auditing Large Language Model (LLM) reasoning across corporate finance, SEC 10-K analysis, and valuation workflows. Designed for RLHF preference ranking, Supervised Fine-Tuning (SFT) data curation, and financial reasoning quality assurance.

---

## 1. Architecture & Scope

General-purpose LLMs frequently produce fluent financial commentary while committing subtle quantitative errors—such as inverting working capital cash flow signs, misapplying anti-dilutive option tranches, or pairing levered cash flows with unlevered discount rates. This repository implements three core components:

1. **Adversarial Financial Reasoning Benchmark (10 Failure Traps):** Edge-case financial scenarios targeting high-frequency LLM failure modes across US GAAP accounting, credit underwriting, and corporate valuation.
2. **Weighted 5-Dimension RLHF Grading Rubric:** Standardized evaluation criteria (0.00–5.00 scale) for scoring candidate model completions during SFT and RLHF preference ranking.
3. **Automated Financial Fact-Checker (`src/verifier.py`):** A deterministic Python audit engine enforcing a strict **0.20% relative error tolerance** against ground-truth financial statement figures.

---

## 2. Adversarial Financial Reasoning Benchmark (10 Core Failure Traps)

| ID | Financial Domain | Adversarial Trap / Prompt Scenario | Typical LLM Failure Mode | Ground-Truth Accounting / Finance Rule |
| :- | :--- | :--- | :--- | :--- |
| **01** | **Diluted EPS (ASC 260)** | Multiple employee stock option tranches with strike prices both below and above current market price. | Sums all options naively or nets out-of-the-money options (creating negative dilution). | Apply **Treasury Stock Method** strictly to in-the-money tranches ($K < P$); exclude anti-dilutive tranches ($K \ge P$). |
| **02** | **Cash Flow Modeling** | Year-over-year expansion in Accounts Receivable and Inventory on the Balance Sheet. | Treats $\Delta \text{NWC}$ increase as a positive cash inflow instead of a cash outflow. | Deduct $\Delta \text{NWC} = \text{NWC}_t - \text{NWC}_{t-1}$; operating asset expansion consumes cash. |
| **03** | **EBITDA Normalization** | 10-K footnote detailing restructuring, legal settlements, and SBC under cash-only prompt rules. | Omits verified non-recurring charges or adds back SBC against explicit prompt exclusions. | Add back verified non-recurring charges; enforce prompt-specific SBC treatment strictly. |
| **04** | **DCF Valuation** | Selecting discount rates for Unlevered Free Cash Flow (FCFF) vs. Levered Free Cash Flow (FCFE). | Discounts FCFE using WACC or discounts FCFF using Cost of Equity ($R_e$). | Pair **FCFF** strictly with **WACC**; pair **FCFE** strictly with **Cost of Equity ($R_e$)**. |
| **05** | **WACC Capital Weights** | Computing capital structure weights when book equity diverges significantly from market cap. | Uses Balance Sheet book equity instead of current market capitalization. | Weight equity by **Market Value ($P \times \text{Shares}$)** and debt by market/carrying principal value. |
| **06** | **Lease Accounting (ASC 842)** | Operating lease liabilities on the Balance Sheet during EV/EBITDA multiple analysis. | Includes Operating Lease Liabilities in Net Debt without adjusting EBITDA to EBITDAR. | Maintain numerator/denominator consistency: exclude operating leases from Debt if using EBITDA. |
| **07** | **Credit Stress-Testing** | Modeling EBITDA impact under a -20% recession revenue contraction. | Assumes constant EBITDA margins, ignoring fixed overhead operating leverage. | Decompose fixed vs. variable OpEx so margins compress realistically under top-line contraction. |
| **08** | **DSCR Underwriting** | Computing Debt Service Coverage Ratio for a senior secured amortizing term loan. | Divides raw EBITDA by interest expense alone, omitting CapEx, cash taxes, and principal amortization. | Use $\text{CFADS} = \text{EBITDA} - \text{CapEx} - \text{Cash Taxes}$ divided by Interest + Mandatory Principal. |
| **09** | **VC Unit Economics** | SaaS LTV/CAC calculation with 75% gross margin and 2.5% monthly logo churn. | Calculates LTV on gross ARPU rather than gross-margin-adjusted ARPU. | Compute $\text{LTV} = \frac{\text{ARPU} \times \text{Gross Margin \%}}{\text{Churn Rate}}$. |
| **10** | **Terminal Growth Sanity** | Estimating Terminal Value via Gordon Growth Model under elevated inflation assumptions. | Accepts user-supplied $g \ge \text{WACC}$, producing negative or undefined Enterprise Value. | Enforce mathematical boundary $g < \text{WACC}$ and anchor $g$ to long-term nominal GDP growth. |

---

## 3. Five-Dimension RLHF Evaluation Rubric

Human domain evaluators and automated reward models score candidate completions across five weighted criteria before inclusion in training datasets:

| Evaluation Dimension | Weight | Score 5.0 (Accept / Gold) | Score 3.0 (Revise) | Score 1.0 (Reject) |
| :--- | :---: | :--- | :--- | :--- |
| **1. Numerical & Arithmetic Grounding** | **35%** | 100% of extracted inputs and computed outputs fall within $\pm 0.20\%$ of ground truth. | Minor rounding drift ($0.20\% - 1.0\%$) with valid underlying setup. | Arithmetic errors, sign inversions, or hallucinated filing figures. |
| **2. Accounting & Valuation Rigor** | **25%** | Explicitly applies US GAAP / IFRS rules (e.g., ASC 260, ASC 842) and pairs cash flows with matching discount rates. | Correct headline formula but omits key adjustment caveats or standard citations. | Fundamental conceptual flaw (e.g., discounting FCFE at WACC, book-value WACC weights). |
| **3. Sign Convention & Edge-Case Discipline** | **20%** | Properly handles $\Delta \text{NWC}$ signs, tax shields, and anti-dilutive option tranches. | Handles standard lines properly but misses one boundary condition. | Inverts cash flow signs or includes anti-dilutive securities in diluted EPS. |
| **4. Constraint & Instruction Adherence** | **10%** | Follows all negative constraints, exclusion rules, and requested output schemas verbatim. | Answers core prompt but includes unrequested adjustments or verbose filler. | Violates explicit prompt exclusions (e.g., adding back excluded SBC). |
| **5. Anti-Sycophancy & Audit Transparency** | **10%** | Rejects flawed user premises explicitly and presents a step-by-step reconciliation trail. | Provides correct final number without showing intermediate reconciliation steps. | Sycophantically validates flawed user assumptions to produce a pleasing response. |

---

## 4. Benchmark Execution & Automated Audit Output

The original normalized FY2025 SEC 10-K reference for **FedEx Corp. (NYSE: FDX)** (`$M`, except per-share data) is retained below. `python run_benchmark.py` now evaluates the 10 adversarial cases in `data/benchmark_cases.json`:

### Deterministic Gold-Standard Ground Truth
- **GAAP EBITDA:** `$9,845.00M` (`$5,559.00M Operating Income + $4,286.00M D&A`)
- **Adjusted EBITDA (Cash-Normalized):** `$10,541.00M` (`$9,845.00M + $584.00M Restructuring + $112.00M Litigation`; `$219.00M SBC` excluded per prompt rule)
- **Net Leverage Ratio:** `1.31x` (`($20,345.00M Total Debt - $6,514.00M Cash) / $10,541.00M Adjusted EBITDA`)
- **Free Cash Flow:** `$3,156.00M` (`$9,845.00M GAAP EBITDA - $1,185.00M Cash Taxes - $5,178.00M CapEx - $326.00M NWC Increase`)
- **Diluted Share Count (ASC 260):** `250.96M` (`248.40M Basic + [8.20M - ($1,492.40M / $264.50)]`; `5.50M` Tranche B options @ `$295.00` strike excluded as anti-dilutive)

### Original FDX Candidate Model Scorecard

| Candidate Model | Checks Passed | Weighted RLHF Score | RLHF Verdict | Detected Failure Modes |
| :--- | :---: | :---: | :--- | :--- |
| **`Model_A_Base`** | `1/5` | **1.57 / 5.00** | **REJECT** | `Adjusted_EBITDA` (Got 9845.0, Exp 10541.0); `Net_Leverage` (Got 1.4, Exp 1.31); `Free_Cash_Flow` (Got 3808.0, Exp 3156.0); `Diluted_Shares` (Got 262.1, Exp 250.96) |
| **`Model_B_Sycophant`** | `2/5` | **2.30 / 5.00** | **REJECT** | `Adjusted_EBITDA` (Got 10760.0, Exp 10541.0); `Net_Leverage` (Got 1.29, Exp 1.31); `Diluted_Shares` (Got 250.32, Exp 250.96) |
| **`Model_C_Aligned`** | `5/5` | **5.00 / 5.00** | **ACCEPT** | `None` |

---

## 5. Usage

```bash
pip install -r requirements.txt
python -m pytest -v
python run_benchmark.py
```

---

## 6. Scope & Limitations

- All 10 adversarial benchmark traps are implemented and verified using structured scalar ground truth and candidate payloads in `data/benchmark_cases.json`.
- Free Cash Flow cases use a standard EBITDA-to-FCF bridge (`GAAP EBITDA - Cash Taxes - CapEx - ΔNWC`) rather than full working-capital sub-ledger schedules. Parsing raw unstructured LLM Markdown tables via regex/AST schema extraction remains out of scope.

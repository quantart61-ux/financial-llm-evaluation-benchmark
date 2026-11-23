Tài liệu này đóng gói trọn bộ 10 bẫy suy luận tài chính chuyên sâu (Adversarial Finance Prompts), thang điểm chấm RLHF 5 chiều chuẩn mực và bảng kết quả kiểm thử tự động.

Markdown
# Financial LLM Reasoning Benchmark & RLHF Evaluation Rubric

An institutional financial domain evaluation suite and automated Python verification engine designed for **AI Training, RLHF Preference Ranking, and Financial Reasoning Auditing**. Built to systematically expose, classify, and score arithmetic hallucinations, accounting standard violations (US GAAP / IFRS), and sycophantic reasoning errors when Large Language Models analyze corporate filings and valuation tasks.

---

## 1. Benchmark Architecture & Purpose

General-purpose LLMs frequently generate fluent financial commentary while committing subtle, high-impact quantitative errors—such as inverting working capital cash flow signs, applying anti-dilutive stock options, or mismatching levered cash flows with unlevered discount rates. This repository provides:

1. **A 10-Category Adversarial Financial Reasoning Benchmark:** Edge-case financial prompts targeting recurring LLM failure modes across Corporate Finance, Accounting Normalization, Credit Underwriting, and Equity Valuation.
2. **A 5-Dimension Weighted RLHF Grading Rubric:** Standardized scoring guidelines (0.00–5.00 scale) for evaluating candidate model completions during Supervised Fine-Tuning (SFT) and Reinforcement Learning from Human Feedback (RLHF).
3. **Automated Python Fact-Checker (`fin_eval_verifier.py`):** A deterministic verification script enforcing a strict **0.20% relative error tolerance** against ground-truth SEC 10-K filing figures.

---

## 2. Adversarial Financial Reasoning Benchmark (10 Core Failure Traps)

| # | Financial Domain | Adversarial Trap / Prompt Scenario | Typical LLM Failure Mode | Ground-Truth Accounting / Finance Rule |
| :- | :--- | :--- | :--- | :--- |
| **01** | **Diluted EPS (ASC 260)** | Computing diluted shares when multiple option tranches exist—some above and some below current stock price. | Sums all options naively or nets out-of-the-money options (creating negative dilution). | Apply **Treasury Stock Method** strictly to in-the-money tranches ($K < P$); exclude anti-dilutive tranches ($K \ge P$). |
| **02** | **Cash Flow Modeling** | Projecting Free Cash Flow when Accounts Receivable and Inventory expand year-over-year. | Adds $\Delta \text{NWC}$ as a positive cash inflow instead of a cash outflow. | Deduct $\Delta \text{NWC} = \text{NWC}_t - \text{NWC}_{t-1}$; operating asset growth consumes cash. |
| **03** | **EBITDA Normalization** | Computing cash-normalized Adjusted EBITDA from a 10-K with restructuring, litigation, and SBC. | Either ignores one-time restructuring charges or blindly adds back SBC against explicit instructions. | Add back verified non-recurring charges; follow prompt-specific treatment for Stock-Based Compensation. |
| **04** | **DCF Valuation** | Valuing equity using Unlevered Free Cash Flow (FCFF) vs. Levered Free Cash Flow (FCFE). | Discounts FCFE using WACC or discounts FCFF using Cost of Equity ($R_e$). | Pair **FCFF** strictly with **WACC**; pair **FCFE** strictly with **Cost of Equity ($R_e$)**. |
| **05** | **WACC Capital Weights** | Computing capital structure weights for a company with low book equity but high market cap. | Uses Balance Sheet Book Value of Equity instead of Market Capitalization. | Compute capital weights using **Market Value of Equity** ($P \times \text{Shares}$) and market/carrying debt. |
| **06** | **Lease Accounting (ASC 842)** | Computing Enterprise Value and EBITDA coverage when operating lease liabilities appear on the Balance Sheet. | Includes Operating Lease Liabilities in Net Debt without adjusting EBITDA to EBITDAR. | Maintain numerator/denominator consistency: exclude operating leases from Debt if using EBITDA. |
| **07** | **Credit Stress-Testing** | Modeling EBITDA impact under a -20% recession revenue shock. | Assumes constant EBITDA margins, ignoring fixed overhead operating leverage. | Decompose fixed vs. variable OpEx so margins compress realistically under revenue contraction. |
| **08** | **DSCR Underwriting** | Computing Debt Service Coverage Ratio for a senior secured term loan. | Divides raw EBITDA by interest expense alone, omitting CapEx, cash taxes, and principal amortization. | Use $\text{CFADS} = \text{EBITDA} - \text{CapEx} - \text{Cash Taxes}$ divided by Interest + Mandatory Amortization. |
| **09** | **VC Unit Economics** | Evaluating SaaS LTV/CAC when gross margin is 75% and monthly logo churn is 2.5%. | Calculates LTV using raw ARPU instead of Gross-Margin-adjusted ARPU. | Compute $\text{LTV} = \frac{\text{ARPU} \times \text{Gross Margin \%}}{\text{Churn Rate}}$. |
| **10** | **Terminal Growth Sanity** | Estimating Terminal Value via Gordon Growth Model in high-inflation scenarios. | Accepts user-suggested $g \ge \text{WACC}$, producing negative or infinite Enterprise Value. | Enforce mathematical boundary $g < \text{WACC}$ and anchor $g$ to long-term nominal GDP growth. |

---

## 3. Five-Dimension RLHF Evaluation Rubric

Human evaluators and automated reward models score candidate completions across five weighted criteria:

| Evaluation Dimension | Weight | Score 5.0 (Gold Standard) | Score 3.0 (Acceptable / Needs Revision) | Score 1.0 (Reject) |
| :--- | :---: | :--- | :--- | :--- |
| **1. Numerical & Arithmetic Grounding** | **35%** | 100% of extracted figures and intermediate/final outputs fall within $\pm 0.20\%$ of ground truth. | Minor rounding drift ($0.20\% - 1.0\%$) with sound underlying setup. | Multiple arithmetic errors, transposed digits, or fabricated filing numbers. |
| **2. Accounting & Valuation Methodology** | **25%** | Explicitly applies correct US GAAP / IFRS rules (e.g., ASC 260, ASC 842) and pairs cash flows with matching discount rates. | Correct headline formula but omits key adjustment caveats or standard citations. | Fundamental conceptual flaw (e.g., discounting FCFE at WACC, book-value WACC weights). |
| **3. Sign Convention & Edge-Case Rigor** | **20%** | Handles $\Delta \text{NWC}$ signs, tax shields, and anti-dilutive tranches flawlessly. | Handles standard items properly but misses one edge-case boundary condition. | Inverts cash flow signs or includes anti-dilutive securities in diluted EPS. |
| **4. Constraint & Instruction Adherence** | **10%** | Follows all negative constraints, exclusion rules, and requested output structures verbatim. | Answers the core question but includes unrequested adjustments or verbose filler. | Ignores explicit prompt constraints (e.g., adding back SBC when prohibited). |
| **5. Anti-Sycophancy & Audit Transparency** | **10%** | Pushes back clearly if user premises violate financial logic; shows transparent step-by-step audit trail. | Provides correct final number without showing intermediate reconciliation steps. | Sycophantically validates flawed user assumptions to produce a pleasing answer. |

---

## 4. Sample Benchmark Execution & Automated Audit Output

Running `python fin_eval_verifier.py` evaluates three candidate LLM completions against a FY2025 SEC 10-K filing excerpt for **Vanguard Logistics & Freight Corp. (NYSE: VLF)**:

### Deterministic Gold-Standard Ground Truth
* **GAAP EBITDA:** `$395.00M` (`$310.00M Operating Income + $85.00M D&A`)
* **Adjusted EBITDA (Cash-Normalized):** `$430.00M` (`$395.00M + $22.00M Restructuring + $13.00M Litigation`; `$18.00M SBC` excluded per prompt rule)
* **Net Leverage Ratio:** `2.23x` (`($1,150.00M Debt - $190.00M Cash) / $430.00M Adjusted EBITDA`)
* **Free Cash Flow:** `$210.00M` (`$395.00M GAAP EBITDA - $58.00M Cash Taxes - $95.00M CapEx - $32.00M NWC Increase`)
* **Diluted Share Count (ASC 260):** `102.25M` (`100.00M Basic + [6.00M - ($150.00M / $40.00)]`; `4.00M` Tranche B options @ `$50.00` strike excluded as anti-dilutive)

### Automated Candidate Model Scorecard

| Candidate Model | Checks Passed | Weighted RLHF Score | RLHF Verdict | Detected Failure Modes |
| :--- | :---: | :---: | :--- | :--- |
| **`LLM_Baseline_v1`** | `1/5` | **1.57 / 5.00** | **REJECT** (Critical Math/GAAP Errors) | `Adjusted_EBITDA` (Got 395.0, Exp 430.0); `Net_Leverage` (Got 2.43, Exp 2.23); `Free_Cash_Flow` (Got 274.0, Exp 210.0); `Diluted_Shares` (Got 110.0, Exp 102.25) |
| **`LLM_Sycophant_v2`** | `2/5` | **2.30 / 5.00** | **REJECT** (Critical Math/GAAP Errors) | `Adjusted_EBITDA` (Got 448.0, Exp 430.0); `Net_Leverage` (Got 2.14, Exp 2.23); `Diluted_Shares` (Got 101.25, Exp 102.25) |
| **`LLM_FinAligned_v3`** | `5/5` | **5.00 / 5.00** | **ACCEPT** (Gold RLHF Sample) | `None (100% Grounded)` |

---

## 5. How to Run

```bash
pip install -r requirements.txt
python fin_eval_verifier.py

---


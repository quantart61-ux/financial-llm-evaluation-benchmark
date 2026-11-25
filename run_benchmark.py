import json
from pathlib import Path

from src.schema import CandidateSubmission
from src.verifier import FinancialEvalVerifier, fdx_fy2025_ground_truth


def main() -> None:
    cases_path = Path(__file__).parent / "data" / "benchmark_cases.json"
    with cases_path.open(encoding="utf-8") as cases_file:
        cases = json.load(cases_file)

    fdx_ground_truth = fdx_fy2025_ground_truth()
    print("FedEx FY2025 reference ground truth:")
    for metric, value in fdx_ground_truth.items():
        print(f"  {metric:<20}: {value:.2f}")
    print()

    rows: list[tuple[str, str, str, str, str, str]] = []
    for case in cases:
        verifier = FinancialEvalVerifier(case["ground_truth"])
        for candidate_data in case["candidates"]:
            report = verifier.verify(CandidateSubmission(**candidate_data))
            failure_summary = ", ".join(report.failures) if report.failures else "None"
            rows.append(
                (
                    case["case_id"],
                    report.candidate_id,
                    f"{report.checks_passed}/{report.total_checks}",
                    f"{report.weighted_score:.2f}",
                    report.verdict,
                    failure_summary,
                )
            )

    headers = ("Case", "Candidate", "Checks", "Score", "Verdict", "Detected failures")
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]
    print(" | ".join(header.ljust(widths[index]) for index, header in enumerate(headers)))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(" | ".join(value.ljust(widths[index]) for index, value in enumerate(row)))


if __name__ == "__main__":
    main()

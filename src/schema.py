from dataclasses import dataclass, field


@dataclass(frozen=True)
class RubricWeights:
    numerical_grounding: float = 0.35
    accounting_rigor: float = 0.25
    edge_case_discipline: float = 0.20
    instruction_adherence: float = 0.10
    audit_transparency: float = 0.10

    def as_mapping(self) -> dict[str, float]:
        return {
            "numerical_grounding": self.numerical_grounding,
            "accounting_rigor": self.accounting_rigor,
            "edge_case_discipline": self.edge_case_discipline,
            "instruction_adherence": self.instruction_adherence,
            "audit_transparency": self.audit_transparency,
        }


@dataclass
class CandidateSubmission:
    candidate_id: str
    metrics: dict[str, float]
    dimension_scores: dict[str, float]


@dataclass
class AuditReport:
    candidate_id: str
    checks_passed: int
    total_checks: int
    weighted_score: float
    verdict: str
    failures: list[str] = field(default_factory=list)

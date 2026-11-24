"""
agentic_controller.py
=====================
A small, auditable agent controller for the loan application.

This is intentionally lightweight: it does not require an LLM or an external
service. The agent observes the application, selects a bounded sequence of
validated tools, adapts the plan for borderline/high-risk cases, and records
its actions for auditability. The deterministic policy engine remains the
final authority for lending decisions.
"""
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List


@dataclass
class AgentState:
    user_data: Dict[str, Any]
    trace: List[Dict[str, Any]] = field(default_factory=list)
    facts: Dict[str, Any] = field(default_factory=dict)

    def record(self, tool: str, summary: str, **details: Any) -> None:
        event = {"tool": tool, "summary": summary}
        if details:
            event["details"] = details
        self.trace.append(event)


class LoanDecisionAgent:
    """Bounded observe-plan-act-reflect controller for loan evaluation."""

    def __init__(self, tools: Dict[str, Callable[..., Any]]):
        self.tools = tools

    def _run(self, state: AgentState, name: str, **kwargs: Any) -> Any:
        result = self.tools[name](**kwargs)
        state.record(name, "completed", result=result if name == "validate_application" else None)
        return result

    def run(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        state = AgentState(user_data=dict(user_data))

        # Observe and validate before selecting any financial tools.
        validation = self._run(state, "validate_application", user_data=user_data)
        state.facts["validation"] = validation
        if not validation["valid"]:
            state.record("reflect", "stopped because required application data is invalid")
            return {
                "decision": "REJECTED",
                "risk_score": 1.0,
                "best_bank": None,
                "rankings": [],
                "eligible_banks": [],
                "ineligible_banks": [],
                "probabilities": {},
                "explanation": "Application cannot be evaluated: " + "; ".join(validation["errors"]),
                "agent_trace": state.trace,
                "agent_mode": "bounded_tool_using_agent",
            }

        # Plan dynamically from observed facts. Borderline cases get a stress test.
        risk = self._run(state, "estimate_risk", user_data=user_data)
        state.facts["risk"] = risk
        risk_score = risk["risk_score"]
        state.record("plan", "selected evaluation path", risk_score=round(risk_score, 4))

        eligibility = self._run(state, "screen_policies", user_data=user_data)
        state.facts["eligibility"] = eligibility

        needs_stress_test = (
            0.45 <= risk_score <= 0.75
            or not eligibility["eligible"]
            or len(eligibility["eligible"]) < len(eligibility["all"])
        )
        if needs_stress_test:
            stress = self._run(state, "stress_test", user_data=user_data)
            state.facts["stress_test"] = stress
            state.record("plan", "added stress test for borderline or constrained profile")

        # Only eligible banks reach the model-backed recommendation tools.
        probabilities = self._run(
            state,
            "predict_bank_approval",
            user_data=user_data,
            eligible_banks=eligibility["eligible"],
        )
        rankings = self._run(
            state,
            "rank_banks",
            user_data=user_data,
            eligible_banks=eligibility["eligible"],
            probabilities=probabilities,
        )

        reflection = self._run(
            state,
            "reflect_decision",
            risk_score=risk_score,
            eligible_banks=eligibility["eligible"],
            stress_test=state.facts.get("stress_test"),
        )
        state.record("reflect", reflection["reason"])
        return {
            "decision": reflection["decision"],
            "risk_score": round(float(risk_score), 4),
            "best_bank": rankings[0].get("bank") if rankings else None,
            "rankings": rankings,
            "eligible_banks": [b["bank"] for b in eligibility["eligible"]],
            "ineligible_banks": eligibility["ineligible"],
            "probabilities": probabilities,
            "agent_trace": state.trace,
            "agent_mode": "bounded_tool_using_agent",
            "reflection": reflection,
        }

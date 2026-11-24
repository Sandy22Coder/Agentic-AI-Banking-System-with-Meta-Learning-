"""
loan_agent.py
=============
End-to-end Agentic Pipeline for Loan Processing.

Pipeline Steps:
  1. Preprocess user input
  2. Compute engineered features
  3. Predict risk (default probability)
  4. Apply bank eligibility rules
  5. Run meta-learning across eligible banks
  6. Rank banks using recommendation engine
  7. Generate human-readable explanation

Returns a structured result dictionary.
"""

"""
loan_agent.py
=============
Public application interface for the bounded, tool-using loan agent.
The Streamlit UI continues to call LoanAgent.process(user_data).
"""
import os
import sys
from typing import Dict, Any

import numpy as np
import pandas as pd

try:
    import joblib
except ImportError:
    joblib = None

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_loader import load_bank_policies
from feature_engineering import preprocess_user_input
from eligibility_rules import check_eligibility, get_eligible_banks, get_ineligible_banks
from meta_learning_model import predict_approval_per_bank
from recommendation_engine import rank_banks
from explainability import generate_explanation
from agentic_controller import LoanDecisionAgent


class HeuristicRiskModel:
    """Transparent fallback used only when trained model artifacts are absent."""

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        row = features.iloc[0]
        dti = float(row.get("dti", 0.0))
        dti_pct = dti if dti > 1 else dti * 100
        credit = float(row.get("credit_score_scaled", 0.5))
        loan_ratio = float(row.get("loan_to_income", 0.0))
        risk = (
            0.45 * min(max(dti_pct / 50.0, 0.0), 1.0)
            + 0.35 * (1.0 - min(max(credit, 0.0), 1.0))
            + 0.20 * min(max(loan_ratio / 3.0, 0.0), 1.0)
        )
        risk = float(np.clip(risk, 0.01, 0.99))
        return np.array([[1.0 - risk, risk]])

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return np.array([int(self.predict_proba(features)[0][1] >= 0.65)])


class HeuristicApprovalModel:
    """Fallback approval scorer with the same sklearn-like interface."""

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        row = features.iloc[0]
        dti = float(row.get("dti", 0.0))
        dti_pct = dti if dti > 1 else dti * 100
        credit = float(row.get("credit_score_scaled", 0.5))
        risk = 0.55 * min(max(dti_pct / 50.0, 0.0), 1.0) + 0.45 * (1 - credit)
        approval = float(np.clip(1 - risk, 0.01, 0.99))
        return np.array([[1.0 - approval, approval]])


class LoanAgent:
    """Bounded agent that plans and executes auditable loan-evaluation tools."""

    RISK_THRESHOLD = 0.65

    def __init__(self, models_dir: str = None, policies_path: str = None):
        if models_dir is None:
            base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            models_dir = os.path.join(base, "models")

        self.using_fallback_models = False
        try:
            if joblib is None:
                raise FileNotFoundError("joblib is not installed")
            self.risk_model = joblib.load(os.path.join(models_dir, "risk_model.pkl"))
            self.approval_model = joblib.load(os.path.join(models_dir, "approval_model.pkl"))
            self.scaler = joblib.load(os.path.join(models_dir, "scaler.pkl"))
        except (OSError, FileNotFoundError, ImportError, ValueError) as exc:
            # The public repository does not ship model binaries.
            print(f"[Agent] Trained models unavailable ({exc}); using safe heuristic fallback.")
            self.risk_model = HeuristicRiskModel()
            self.approval_model = HeuristicApprovalModel()
            self.scaler = None
            self.using_fallback_models = True

        self.bank_policies = load_bank_policies(policies_path)

    @staticmethod
    def _validate_application(user_data: Dict[str, Any]) -> Dict[str, Any]:
        required = ("income", "credit_score", "loan_amount", "emi", "employment_type")
        errors = [f"missing {key}" for key in required if key not in user_data]
        for key in ("income", "credit_score", "loan_amount", "emi"):
            if key in user_data:
                try:
                    if float(user_data[key]) < 0:
                        errors.append(f"{key} cannot be negative")
                except (TypeError, ValueError):
                    errors.append(f"{key} must be numeric")
        if "income" in user_data and float(user_data.get("income", 0)) == 0:
            errors.append("income must be greater than zero")
        return {"valid": not errors, "errors": errors}

    def _tools(self) -> Dict[str, Any]:
        def estimate_risk(user_data: Dict[str, Any]) -> Dict[str, Any]:
            features = preprocess_user_input(user_data, scaler=self.scaler)
            probability = float(self.risk_model.predict_proba(features)[0][1])
            return {"risk_score": probability, "model": type(self.risk_model).__name__}

        def screen_policies(user_data: Dict[str, Any]) -> Dict[str, Any]:
            income = float(user_data["income"])
            monthly_income = income / 12
            dti = float(user_data["emi"]) / monthly_income if monthly_income else 1.0
            results = check_eligibility(
                credit_score=float(user_data["credit_score"]),
                income=income,
                dti=dti,
                bank_policies=self.bank_policies,
            )
            return {
                "all": results,
                "eligible": get_eligible_banks(results),
                "ineligible": get_ineligible_banks(results),
            }

        def stress_test(user_data: Dict[str, Any]) -> Dict[str, Any]:
            # Check whether eligibility depends on a fragile income/EMI boundary.
            stressed = dict(user_data)
            stressed["income"] = float(user_data["income"]) * 0.90
            stressed["emi"] = float(user_data["emi"]) * 1.10
            monthly_income = stressed["income"] / 12
            dti = stressed["emi"] / monthly_income if monthly_income else 1.0
            results = check_eligibility(
                credit_score=float(stressed["credit_score"]),
                income=stressed["income"],
                dti=dti,
                bank_policies=self.bank_policies,
            )
            return {"eligible_count_under_stress": len(get_eligible_banks(results))}

        def predict_bank_approval(user_data: Dict[str, Any], eligible_banks):
            features = preprocess_user_input(user_data, scaler=self.scaler)
            return predict_approval_per_bank(
                user_features=features,
                eligible_banks=eligible_banks,
                approval_model=self.approval_model,
            )

        def rank_options(user_data: Dict[str, Any], eligible_banks, probabilities):
            return rank_banks(
                approval_probabilities=probabilities,
                eligible_banks=eligible_banks,
                loan_amount=float(user_data["loan_amount"]),
                income=float(user_data["income"]),
            )

        def reflect_decision(risk_score, eligible_banks, stress_test=None):
            if not eligible_banks:
                return {"decision": "REJECTED", "reason": "No bank policy accepted the application."}
            if risk_score > self.RISK_THRESHOLD:
                return {"decision": "REJECTED", "reason": "Risk model exceeded the policy threshold."}
            if stress_test and stress_test["eligible_count_under_stress"] == 0:
                return {"decision": "REJECTED", "reason": "The application failed the affordability stress test."}
            return {"decision": "APPROVED", "reason": "Policy checks, risk threshold, and agent reflection passed."}

        return {
            "validate_application": self._validate_application,
            "estimate_risk": estimate_risk,
            "screen_policies": screen_policies,
            "stress_test": stress_test,
            "predict_bank_approval": predict_bank_approval,
            "rank_banks": rank_options,
            "reflect_decision": reflect_decision,
        }

    def process(self, user_data: dict) -> Dict:
        agent = LoanDecisionAgent(self._tools())
        result = agent.run(user_data)
        eligible = [{"bank": b} for b in result.get("eligible_banks", [])]
        result["explanation"] = generate_explanation(
            user_data=user_data,
            decision=result["decision"],
            risk_score=result["risk_score"],
            eligible_banks=eligible,
            ineligible_banks=result.get("ineligible_banks", []),
            best_bank=result.get("rankings", [{}])[0] if result.get("rankings") else None,
            rankings=result.get("rankings", []),
        )
        if self.using_fallback_models:
            result["explanation"] += "\n\nNote: trained model files were not found; a transparent heuristic fallback was used."
        return result


if __name__ == "__main__":
    agent = LoanAgent()
    print(agent.process({
        "income": 75000,
        "credit_score": 720,
        "loan_amount": 25000,
        "emi": 1500,
        "employment_type": "Salaried",
    }))

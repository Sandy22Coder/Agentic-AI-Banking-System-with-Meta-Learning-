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

import os
import sys
import joblib
import pandas as pd
from typing import Dict

# Ensure src is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import load_bank_policies
from feature_engineering import preprocess_user_input, FEATURE_COLUMNS
from eligibility_rules import check_eligibility, get_eligible_banks, get_ineligible_banks
from meta_learning_model import predict_approval_per_bank
from recommendation_engine import rank_banks, get_best_bank
from explainability import generate_explanation


class LoanAgent:
    """
    Agentic AI system that orchestrates the full loan evaluation pipeline.
    
    Attributes
    ----------
    risk_model : trained model
        Random Forest for default risk prediction.
    approval_model : trained model
        Gradient Boosting for approval prediction.
    scaler : StandardScaler
        Fitted feature scaler.
    bank_policies : pd.DataFrame
        Bank policy dataset.
    """
    
    def __init__(self, models_dir: str = None, policies_path: str = None):
        """
        Initialize the agent by loading models and bank policies.
        """
        if models_dir is None:
            base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            models_dir = os.path.join(base, 'models')
        
        # Load trained models
        print("[Agent] Loading models...")
        self.risk_model = joblib.load(os.path.join(models_dir, 'risk_model.pkl'))
        self.approval_model = joblib.load(os.path.join(models_dir, 'approval_model.pkl'))
        self.scaler = joblib.load(os.path.join(models_dir, 'scaler.pkl'))
        print("[Agent] Models loaded successfully.")
        
        # Load bank policies
        self.bank_policies = load_bank_policies(policies_path)
    
    def process(self, user_data: dict) -> Dict:
        """
        Run the full agentic pipeline on a user's loan application.
        
        Parameters
        ----------
        user_data : dict
            Keys: income, credit_score, loan_amount, emi, employment_type
            
        Returns
        -------
        dict
            Complete result with: decision, risk_score, best_bank,
            rankings, eligible_banks, explanation, probabilities
        """
        print("\n" + "=" * 70)
        print("  [*] AGENTIC LOAN PROCESSING PIPELINE")
        print("=" * 70)
        
        # ------------------------------------------------------------------
        # Step 1: Preprocess Input
        # ------------------------------------------------------------------
        print("\n[Step 1] Preprocessing user input...")
        income = user_data.get('income', 0)
        credit_score = user_data.get('credit_score', 0)
        loan_amount = user_data.get('loan_amount', 0)
        emi = user_data.get('emi', 0)
        emp_type = user_data.get('employment_type', 'Salaried')
        
        print(f"  Income: ${income:,.0f} | Credit: {credit_score} | "
              f"Loan: ${loan_amount:,.0f} | EMI: ${emi:,.0f} | Emp: {emp_type}")
        
        # ------------------------------------------------------------------
        # Step 2: Compute Features
        # ------------------------------------------------------------------
        print("\n[Step 2] Computing ratio-based features...")
        features_df = preprocess_user_input(user_data, scaler=self.scaler)
        features_unscaled = preprocess_user_input(user_data, scaler=None)
        
        monthly_income = income / 12
        dti_ratio = (emi / monthly_income) if monthly_income > 0 else 1.0
        loan_to_income = loan_amount / (income + 1)
        emi_ratio = emi / (income + 1)
        
        print(f"  DTI: {dti_ratio:.4f} | Loan/Income: {loan_to_income:.4f} | EMI Ratio: {emi_ratio:.6f}")
        
        # ------------------------------------------------------------------
        # Step 3: Predict Risk
        # ------------------------------------------------------------------
        print("\n[Step 3] Predicting default risk...")
        risk_prob = self.risk_model.predict_proba(features_df)[0][1]
        risk_class = self.risk_model.predict(features_df)[0]
        print(f"  Risk Score: {risk_prob:.4f} | Risk Class: {'HIGH' if risk_class == 1 else 'LOW'}")
        
        # ------------------------------------------------------------------
        # Step 4: Apply Bank Eligibility Rules
        # ------------------------------------------------------------------
        print("\n[Step 4] Checking bank eligibility...")
        eligibility_results = check_eligibility(
            credit_score=credit_score,
            income=income,
            dti=dti_ratio,
            bank_policies=self.bank_policies
        )
        eligible = get_eligible_banks(eligibility_results)
        ineligible = get_ineligible_banks(eligibility_results)
        
        # ------------------------------------------------------------------
        # Decision logic
        # ------------------------------------------------------------------
        RISK_THRESHOLD = 0.65
        
        if not eligible or risk_prob > RISK_THRESHOLD:
            # REJECTED
            decision = 'REJECTED'
            explanation = generate_explanation(
                user_data=user_data,
                decision=decision,
                risk_score=risk_prob,
                eligible_banks=eligible,
                ineligible_banks=ineligible
            )
            
            return {
                'decision': decision,
                'risk_score': round(float(risk_prob), 4),
                'best_bank': None,
                'rankings': [],
                'eligible_banks': [b['bank'] for b in eligible],
                'probabilities': {},
                'explanation': explanation,
            }
        
        # ------------------------------------------------------------------
        # Step 5: Meta-Learning -- Predict per-bank approval probability
        # ------------------------------------------------------------------
        print("\n[Step 5] Running meta-learning simulation...")
        probabilities = predict_approval_per_bank(
            user_features=features_df,
            eligible_banks=eligible,
            approval_model=self.approval_model
        )
        
        # ------------------------------------------------------------------
        # Step 6: Rank Banks
        # ------------------------------------------------------------------
        print("\n[Step 6] Ranking banks...")
        rankings = rank_banks(
            approval_probabilities=probabilities,
            eligible_banks=eligible,
            loan_amount=loan_amount,
            income=income
        )
        best = get_best_bank(rankings)
        
        # ------------------------------------------------------------------
        # Step 7: Generate Explanation
        # ------------------------------------------------------------------
        print("\n[Step 7] Generating explanation...")
        decision = 'APPROVED'
        explanation = generate_explanation(
            user_data=user_data,
            decision=decision,
            risk_score=risk_prob,
            eligible_banks=eligible,
            ineligible_banks=ineligible,
            best_bank=best,
            rankings=rankings
        )
        
        return {
            'decision': decision,
            'risk_score': round(float(risk_prob), 4),
            'best_bank': best.get('bank') if best else None,
            'rankings': rankings,
            'eligible_banks': [b['bank'] for b in eligible],
            'probabilities': probabilities,
            'explanation': explanation,
        }


# ---- CLI test ----
if __name__ == '__main__':
    agent = LoanAgent()
    
    result = agent.process({
        'income': 75000,
        'credit_score': 720,
        'loan_amount': 25000,
        'emi': 1500,
        'employment_type': 'Salaried'
    })
    
    print("\n\n" + result['explanation'])
    print(f"\nDecision: {result['decision']}")
    print(f"Best Bank: {result['best_bank']}")

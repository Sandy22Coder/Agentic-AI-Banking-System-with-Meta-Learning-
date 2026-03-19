"""
eligibility_rules.py
====================
Rule engine to check user eligibility against each bank's policy.

Rules:
  - credit_score >= bank's min_credit_score
  - income       >= bank's min_income
  - dti          <= bank's max_dti
"""

import pandas as pd
from typing import List, Dict


def check_eligibility(
    credit_score: float,
    income: float,
    dti: float,
    bank_policies: pd.DataFrame
) -> List[Dict]:
    """
    Check user eligibility against all bank policies.
    
    Parameters
    ----------
    credit_score : float
        User's credit score.
    income : float
        User's annual income.
    dti : float
        User's debt-to-income ratio (as a decimal, e.g. 0.35).
    bank_policies : pd.DataFrame
        DataFrame with columns: bank, min_credit_score, min_income, max_dti, interest_rate
    
    Returns
    -------
    list of dict
        Each dict has: bank, eligible (bool), reasons (list of failure reasons),
        interest_rate, and the bank's policy parameters.
    """
    results = []
    
    # Normalize DTI: if user's DTI is > 1, assume it's a percentage and convert
    dti_decimal = dti / 100 if dti > 1 else dti
    
    for _, policy in bank_policies.iterrows():
        reasons = []
        eligible = True
        
        if credit_score < policy['min_credit_score']:
            eligible = False
            reasons.append(
                f"Credit score {credit_score:.0f} < required {policy['min_credit_score']:.0f}"
            )
        
        if income < policy['min_income']:
            eligible = False
            reasons.append(
                f"Annual income ${income:,.0f} < required ${policy['min_income']:,.0f}"
            )
        
        if dti_decimal > policy['max_dti']:
            eligible = False
            reasons.append(
                f"DTI {dti_decimal:.2%} > max allowed {policy['max_dti']:.2%}"
            )
        
        results.append({
            'bank': policy['bank'],
            'eligible': eligible,
            'reasons': reasons,
            'interest_rate': policy['interest_rate'],
            'min_credit_score': policy['min_credit_score'],
            'min_income': policy['min_income'],
            'max_dti': policy['max_dti'],
        })
    
    eligible_count = sum(1 for r in results if r['eligible'])
    print(f"[Rules] Eligible banks: {eligible_count}/{len(results)}")
    return results


def get_eligible_banks(results: List[Dict]) -> List[Dict]:
    """Filter to only eligible banks."""
    return [r for r in results if r['eligible']]


def get_ineligible_banks(results: List[Dict]) -> List[Dict]:
    """Filter to only ineligible banks."""
    return [r for r in results if not r['eligible']]


# ---- Quick test ----
if __name__ == '__main__':
    from data_loader import load_bank_policies
    
    policies = load_bank_policies()
    results = check_eligibility(
        credit_score=720,
        income=60000,
        dti=0.30,
        bank_policies=policies
    )
    
    for r in results:
        status = "[+] Eligible" if r['eligible'] else "[-] Ineligible"
        print(f"  {r['bank']:20s} -> {status}")
        if r['reasons']:
            for reason in r['reasons']:
                print(f"    - {reason}")

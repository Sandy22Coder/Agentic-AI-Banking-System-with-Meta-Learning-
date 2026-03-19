"""
recommendation_engine.py
========================
Ranks eligible banks using a weighted scoring formula.

Score = 0.5 × approval_probability
     + 0.3 × (1 / interest_rate) [normalized]
     + 0.2 × affordability_score
"""

import pandas as pd
from typing import Dict, List, Tuple


def compute_affordability(loan_amount: float, income: float, 
                          interest_rate: float) -> float:
    """
    Compute affordability score in [0, 1].
    
    Higher is better — measures how easily the borrower can
    service the loan at the given interest rate.
    """
    if income <= 0:
        return 0.0
    
    # Annual payment estimation (simple interest, 3-year term)
    annual_payment = loan_amount * (1 + interest_rate) / 3
    payment_ratio = annual_payment / income
    
    # Affordability: 1 when payment is negligible, 0 when >= income
    affordability = max(0, 1 - payment_ratio)
    return round(affordability, 4)


def rank_banks(
    approval_probabilities: Dict[str, float],
    eligible_banks: List[Dict],
    loan_amount: float,
    income: float
) -> List[Dict]:
    """
    Rank banks using the weighted scoring formula.
    
    Parameters
    ----------
    approval_probabilities : dict
        bank_name → approval probability.
    eligible_banks : list of dict
        Eligible banks with interest rates.
    loan_amount : float
        Requested loan amount.
    income : float
        Annual income.
        
    Returns
    -------
    list of dict
        Sorted list (best first) with bank, score, approval_prob, 
        interest_rate, affordability.
    """
    rankings = []
    
    # Normalize interest rate component: 1/interest_rate
    # Scale to [0, 1] across all eligible banks
    interest_rates = [b['interest_rate'] for b in eligible_banks 
                      if b['bank'] in approval_probabilities]
    if interest_rates:
        inv_rates = [1 / r for r in interest_rates]
        max_inv = max(inv_rates)
        min_inv = min(inv_rates)
        rate_range = max_inv - min_inv if max_inv != min_inv else 1
    
    for bank_info in eligible_banks:
        bank_name = bank_info['bank']
        if bank_name not in approval_probabilities:
            continue
        
        approval_prob = approval_probabilities[bank_name]
        interest_rate = bank_info['interest_rate']
        
        # Normalized inverse interest rate
        inv_rate = 1 / interest_rate
        if rate_range > 0:
            interest_score = (inv_rate - min_inv) / rate_range
        else:
            interest_score = 0.5
        
        # Affordability
        affordability = compute_affordability(loan_amount, income, interest_rate)
        
        # Weighted score
        score = (
            0.5 * approval_prob +
            0.3 * interest_score +
            0.2 * affordability
        )
        
        rankings.append({
            'bank': bank_name,
            'score': round(score, 4),
            'approval_probability': round(approval_prob, 4),
            'interest_rate': interest_rate,
            'interest_score': round(interest_score, 4),
            'affordability': round(affordability, 4),
        })
    
    # Sort by score descending
    rankings.sort(key=lambda x: x['score'], reverse=True)
    
    print(f"\n[Ranking] Bank Rankings:")
    for i, r in enumerate(rankings, 1):
        print(f"  {i}. {r['bank']:20s} -> Score: {r['score']:.4f} "
              f"(Approval: {r['approval_probability']:.4f}, "
              f"Rate: {r['interest_rate']:.3f}, "
              f"Afford: {r['affordability']:.4f})")
    
    return rankings


def get_best_bank(rankings: List[Dict]) -> Dict:
    """Return the top-ranked bank."""
    if rankings:
        return rankings[0]
    return {}


# ---- Quick test ----
if __name__ == '__main__':
    # Simulated test
    probs = {'HDFC': 0.85, 'SBI': 0.72, 'ICICI': 0.78}
    banks = [
        {'bank': 'HDFC', 'interest_rate': 0.09},
        {'bank': 'SBI', 'interest_rate': 0.10},
        {'bank': 'ICICI', 'interest_rate': 0.095},
    ]
    rankings = rank_banks(probs, banks, 500000, 80000)
    print(f"\nBest bank: {get_best_bank(rankings)}")

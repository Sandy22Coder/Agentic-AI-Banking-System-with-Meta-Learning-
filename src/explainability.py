"""
explainability.py
=================
Generates human-readable explanations for loan decisions.

- If rejected: highlights which factors failed (low credit, high DTI, etc.)
- If approved: highlights strong features and best bank recommendation
"""

from typing import Dict, List


def _assess_feature_strength(user_data: dict) -> Dict[str, str]:
    """
    Assess each feature as 'strong', 'moderate', or 'weak'.
    """
    assessments = {}
    
    # Credit Score assessment
    cs = user_data.get('credit_score', 0)
    if cs >= 750:
        assessments['credit_score'] = 'strong'
    elif cs >= 680:
        assessments['credit_score'] = 'moderate'
    else:
        assessments['credit_score'] = 'weak'
    
    # Income assessment (relative to loan)
    income = user_data.get('income', 0)
    loan = user_data.get('loan_amount', 0)
    if income > 0:
        ratio = loan / income
        if ratio < 0.5:
            assessments['income'] = 'strong'
        elif ratio < 1.5:
            assessments['income'] = 'moderate'
        else:
            assessments['income'] = 'weak'
    else:
        assessments['income'] = 'weak'
    
    # DTI assessment
    emi = user_data.get('emi', 0)
    monthly_inc = income / 12 if income > 0 else 1
    dti = emi / monthly_inc if monthly_inc > 0 else 1
    if dti < 0.30:
        assessments['dti'] = 'strong'
    elif dti < 0.45:
        assessments['dti'] = 'moderate'
    else:
        assessments['dti'] = 'weak'
    
    # Employment assessment
    emp = user_data.get('employment_type', 'Unknown')
    if emp in ('Salaried', 'Retired'):
        assessments['employment'] = 'strong'
    elif emp == 'Self-Employed':
        assessments['employment'] = 'moderate'
    else:
        assessments['employment'] = 'weak'
    
    return assessments


def generate_explanation(
    user_data: dict,
    decision: str,
    risk_score: float,
    eligible_banks: List[Dict],
    ineligible_banks: List[Dict],
    best_bank: Dict = None,
    rankings: List[Dict] = None
) -> str:
    """
    Generate a comprehensive explanation for the loan decision.
    
    Parameters
    ----------
    user_data : dict
        Original user inputs.
    decision : str
        'APPROVED' or 'REJECTED'.
    risk_score : float
        Predicted default risk (0-1).
    eligible_banks : list
        Banks where user is eligible.
    ineligible_banks : list
        Banks where user is not eligible.
    best_bank : dict, optional
        Top-ranked bank info.
    rankings : list, optional
        Full bank rankings.
        
    Returns
    -------
    str
        Human-readable explanation.
    """
    lines = []
    assessments = _assess_feature_strength(user_data)
    
    lines.append("=" * 60)
    lines.append("  LOAN DECISION EXPLANATION")
    lines.append("=" * 60)
    
    # ------------------------------------------------------------------
    # Decision summary
    # ------------------------------------------------------------------
    if decision == 'APPROVED':
        lines.append(f"\n[+] DECISION: APPROVED")
        lines.append(f"   Risk Score: {risk_score:.2%} (probability of default)")
        
        if best_bank:
            lines.append(f"\n>>> BEST BANK: {best_bank.get('bank', 'N/A')}")
            lines.append(f"   Overall Score: {best_bank.get('score', 0):.4f}")
            lines.append(f"   Approval Probability: {best_bank.get('approval_probability', 0):.2%}")
            lines.append(f"   Interest Rate: {best_bank.get('interest_rate', 0):.2%}")
        
        # Highlight strong features
        lines.append(f"\n--- PROFILE STRENGTHS:")
        strong = [k for k, v in assessments.items() if v == 'strong']
        moderate = [k for k, v in assessments.items() if v == 'moderate']
        
        if strong:
            for feat in strong:
                lines.append(f"   [+] {feat.replace('_', ' ').title()}: Strong")
        if moderate:
            for feat in moderate:
                lines.append(f"   ~ {feat.replace('_', ' ').title()}: Moderate")
        
        lines.append(f"\n   Eligible at {len(eligible_banks)} out of "
                      f"{len(eligible_banks) + len(ineligible_banks)} banks.")
    
    else:  # REJECTED
        lines.append(f"\n[-] DECISION: REJECTED")
        lines.append(f"   Risk Score: {risk_score:.2%} (probability of default)")
        
        # Highlight weak features
        lines.append(f"\n[!] REJECTION REASONS:")
        weak = [k for k, v in assessments.items() if v == 'weak']
        
        if weak:
            for feat in weak:
                label = feat.replace('_', ' ').title()
                if feat == 'credit_score':
                    lines.append(f"   [-] Low {label}: {user_data.get('credit_score', 'N/A')}")
                    lines.append(f"     -> Improve to 700+ for more bank options")
                elif feat == 'income':
                    lines.append(f"   [-] {label} insufficient for requested loan amount")
                    lines.append(f"     -> Consider a smaller loan or increase income")
                elif feat == 'dti':
                    lines.append(f"   [-] High Debt-to-Income ratio")
                    lines.append(f"     -> Reduce existing debts before applying")
                elif feat == 'employment':
                    lines.append(f"   [-] Employment type: {user_data.get('employment_type', 'Unknown')}")
                    lines.append(f"     -> Stable employment improves eligibility")
        
        if not eligible_banks:
            lines.append(f"\n   Not eligible at any of the {len(ineligible_banks)} banks evaluated.")
        else:
            lines.append(f"\n   Eligible at {len(eligible_banks)} banks, but risk score is too high.")
        
        # Show specific bank rejection reasons
        if ineligible_banks:
            lines.append(f"\n   Bank-specific reasons:")
            for bank in ineligible_banks[:3]:  # Show top 3
                lines.append(f"   * {bank['bank']}:")
                for reason in bank.get('reasons', []):
                    lines.append(f"     - {reason}")
    
    lines.append("\n" + "=" * 60)
    return "\n".join(lines)


# ---- Quick test ----
if __name__ == '__main__':
    explanation = generate_explanation(
        user_data={'income': 80000, 'credit_score': 720, 'loan_amount': 50000,
                   'emi': 2000, 'employment_type': 'Salaried'},
        decision='APPROVED',
        risk_score=0.15,
        eligible_banks=[{'bank': 'HDFC'}, {'bank': 'SBI'}],
        ineligible_banks=[{'bank': 'Kotak', 'reasons': ['Credit score 720 < required 750']}],
        best_bank={'bank': 'HDFC', 'score': 0.85, 'approval_probability': 0.90,
                   'interest_rate': 0.09}
    )
    print(explanation)

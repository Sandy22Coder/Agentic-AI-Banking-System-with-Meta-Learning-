"""
meta_learning_model.py
======================
Simulates meta-learning by treating each bank as a separate "task".

For each eligible bank:
  - Combines user features with bank-specific policy features
  - Predicts an approval probability using the trained approval model
  - This simulates task-specific adaptation: different banks (tasks) 
    yield different predictions based on their unique policy context.
"""

import numpy as np
import pandas as pd
from typing import Dict, List


def compute_meta_features(user_features: pd.DataFrame,
                          bank_policy: dict) -> pd.DataFrame:
    """
    Create task-specific features by combining user ratio features with 
    bank policy parameters.
    
    The meta-learning concept:
      Each bank is a "task". We adapt the feature representation
      per-task by incorporating bank-specific context, mimicking
      how meta-learning adapts a base model to new tasks.
    
    Now operates on ratio-based features (currency-independent).
    
    Parameters
    ----------
    user_features : pd.DataFrame
        Single-row DataFrame with scaled ratio features.
    bank_policy : dict
        Bank policy with min_credit_score, max_dti, interest_rate, etc.
        
    Returns
    -------
    pd.DataFrame
        Augmented features (same columns, adjusted values).
    """
    meta = user_features.copy()
    
    # --- Adapt ratio features relative to bank's policy ---
    
    # Credit score margin: bias towards banks with lower thresholds
    if 'credit_score_scaled' in meta.columns:
        threshold_scaled = (bank_policy['min_credit_score'] - 300) / 550
        meta['credit_score_scaled'] = meta['credit_score_scaled'].values + (0.1 * (1 - threshold_scaled))
    
    # DTI adjustment: banks with higher max_dti are more lenient
    if 'dti' in meta.columns:
        dti_factor = 1 - (0.05 * bank_policy['max_dti'])
        meta['dti'] = meta['dti'].values * dti_factor
    
    # Loan-to-income ratio: adjust based on bank tolerance
    if 'loan_to_income' in meta.columns:
        # Banks with higher DTI tolerance also tolerate higher loan ratios
        lti_factor = 1 - (0.03 * (bank_policy['max_dti'] - 0.4))
        meta['loan_to_income'] = meta['loan_to_income'].values * lti_factor
    
    # EMI ratio: adjust based on interest rate (lower rate = more affordable)
    if 'emi_ratio' in meta.columns:
        rate_factor = 1 - (0.5 * (0.10 - bank_policy['interest_rate']))
        meta['emi_ratio'] = meta['emi_ratio'].values * rate_factor
    
    return meta


def predict_approval_per_bank(
    user_features: pd.DataFrame,
    eligible_banks: List[Dict],
    approval_model
) -> Dict[str, float]:
    """
    Predict approval probability for each eligible bank using
    meta-learning simulation.
    
    Parameters
    ----------
    user_features : pd.DataFrame
        Scaled user features (single row).
    eligible_banks : list of dict
        Eligible banks with policy parameters.
    approval_model : trained sklearn model
        Approval prediction model.
    
    Returns
    -------
    dict
        bank_name → approval_probability
    """
    probabilities = {}
    
    for bank_info in eligible_banks:
        bank_name = bank_info['bank']
        
        # Create task-specific features for this bank
        meta_features = compute_meta_features(user_features, bank_info)
        
        # Predict approval probability
        prob = approval_model.predict_proba(meta_features)[0][1]
        
        # Apply bank-specific calibration
        # (simulates meta-learning task adaptation)
        interest_penalty = bank_info['interest_rate'] * 0.5  # Lower rate banks are stricter
        calibrated_prob = np.clip(prob - interest_penalty + 0.05, 0.01, 0.99)
        
        probabilities[bank_name] = round(float(calibrated_prob), 4)
        
        print(f"[MetaLearning] {bank_name:20s} -> "
              f"raw: {prob:.4f}, calibrated: {calibrated_prob:.4f}")
    
    return probabilities


# ---- Quick test ----
if __name__ == '__main__':
    print("Meta-learning module loaded. Use with trained models.")

"""
feature_engineering.py
======================
Creates targets, engineers RATIO-BASED features, and prepares train/test splits.

Currency-Independent Design:
  Models rely on ratios (not raw $ values) so the system works across
  regions (US, India, etc.) without retraining.

Targets:
  - risk_target:     Charged Off / Default -> 1, Fully Paid -> 0
  - approval_target: Fully Paid -> 1, Charged Off / Default -> 0

Engineered ratio features:
  - dti                = existing Lending Club DTI (monthly_debt / monthly_income * 100)
  - emi_ratio          = installment / annual_inc   (monthly EMI burden relative to income)
  - loan_to_income     = loan_amnt / annual_inc     (loan size relative to income)
  - credit_score_scaled = (credit_score - 300) / 550 (normalized FICO)
  - emp_length         = employment duration in years
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# RATIO-BASED feature set (currency-independent)
# Removed raw loan_amnt and annual_inc; replaced with ratios.
# ---------------------------------------------------------------------------
FEATURE_COLUMNS = [
    'dti',                  # Lending Club DTI (percentage)
    'emi_ratio',            # installment / annual_inc
    'loan_to_income',       # loan_amnt / annual_inc
    'credit_score_scaled',  # normalized FICO [0, 1]
    'emp_length',           # years of employment
]


def create_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create binary classification targets.
    
    - risk_target:     1 if Charged Off or Default, else 0
    - approval_target: 1 if Fully Paid, else 0
    """
    df = df.copy()
    df['risk_target'] = df['loan_status'].apply(
        lambda s: 1 if s in ('Charged Off', 'Default') else 0
    )
    df['approval_target'] = df['loan_status'].apply(
        lambda s: 1 if s == 'Fully Paid' else 0
    )
    print(f"[FeatureEng] Risk target distribution:\n{df['risk_target'].value_counts().to_dict()}")
    print(f"[FeatureEng] Approval target distribution:\n{df['approval_target'].value_counts().to_dict()}")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer ratio-based features from raw columns.
    
    All financial ratios use safe division (+1 guard) and inf replacement.
    """
    df = df.copy()

    # 1. EMI-to-Income ratio (installment is monthly; annual_inc is yearly)
    df['emi_ratio'] = df['installment'] / (df['annual_inc'] + 1)

    # 2. Loan-to-Income ratio
    df['loan_to_income'] = df['loan_amnt'] / (df['annual_inc'] + 1)

    # 3. Credit score scaled to [0, 1]  (FICO 300-850)
    df['credit_score_scaled'] = ((df['credit_score'] - 300) / 550).clip(0, 1)

    # 4. dti already exists from dataset (in percentage form)
    #    -- keep as-is

    # 5. Replace any inf / -inf with 0, then fill remaining NaN with 0
    for col in ['emi_ratio', 'loan_to_income']:
        df[col] = df[col].replace([np.inf, -np.inf], 0).fillna(0)

    print(f"[FeatureEng] Ratio features engineered: "
          f"emi_ratio mean={df['emi_ratio'].mean():.6f}, "
          f"loan_to_income mean={df['loan_to_income'].mean():.4f}")
    return df


def prepare_data(df: pd.DataFrame, target_col: str = 'risk_target',
                 test_size: float = 0.2, random_state: int = 42):
    """
    Prepare train/test split with StandardScaler normalization.
    
    Returns
    -------
    X_train, X_test, y_train, y_test, scaler
    """
    X = df[FEATURE_COLUMNS].copy()
    y = df[target_col].copy()

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Scale all features to comparable range
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=FEATURE_COLUMNS,
        index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=FEATURE_COLUMNS,
        index=X_test.index
    )

    print(f"[FeatureEng] Train: {X_train_scaled.shape}, Test: {X_test_scaled.shape}")
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def preprocess_user_input(user_data: dict, scaler=None) -> pd.DataFrame:
    """
    Preprocess a single user's Streamlit input into ratio-based features
    compatible with the trained models.
    
    Parameters
    ----------
    user_data : dict
        Keys: income, credit_score, loan_amount, emi, employment_type
    scaler : StandardScaler, optional
        Fitted scaler. If None, returns unscaled features.
    
    Returns
    -------
    pd.DataFrame
        Single-row DataFrame with FEATURE_COLUMNS.
    """
    # Map employment type to approximate years
    emp_type_map = {
        'Salaried': 5,
        'Self-Employed': 7,
        'Unemployed': 0,
        'Student': 0,
        'Retired': 10,
    }

    income = user_data.get('income', 0)
    credit_score = user_data.get('credit_score', 650)
    loan_amount = user_data.get('loan_amount', 0)
    emi = user_data.get('emi', 0)
    emp_type = user_data.get('employment_type', 'Salaried')

    # Compute ratio-based features (currency-independent)
    monthly_income = income / 12
    dti = (emi / monthly_income * 100) if monthly_income > 0 else 100  # as percentage
    emi_ratio = emi / (income + 1)          # monthly EMI / annual income
    loan_to_income = loan_amount / (income + 1)  # loan / annual income

    row = {
        'dti': dti,
        'emi_ratio': emi_ratio,
        'loan_to_income': loan_to_income,
        'credit_score_scaled': np.clip((credit_score - 300) / 550, 0, 1),
        'emp_length': emp_type_map.get(emp_type, 5),
    }

    df = pd.DataFrame([row], columns=FEATURE_COLUMNS)

    if scaler is not None:
        df = pd.DataFrame(
            scaler.transform(df),
            columns=FEATURE_COLUMNS
        )

    return df


# ---- Quick test ----
if __name__ == '__main__':
    from data_loader import load_dataset

    df = load_dataset(sample_frac=0.05)
    df = create_targets(df)
    df = engineer_features(df)
    print("\nRatio-based features sample:")
    print(df[FEATURE_COLUMNS + ['risk_target', 'approval_target']].head(10))

    X_train, X_test, y_train, y_test, scaler = prepare_data(df, 'risk_target')
    print("\nScaled features sample:")
    print(X_train.head())

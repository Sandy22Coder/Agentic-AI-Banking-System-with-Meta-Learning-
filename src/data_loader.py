"""
data_loader.py
==============
Handles loading and cleaning of the Lending Club dataset.
- Selects relevant columns
- Filters valid loan statuses (Fully Paid, Charged Off, Default)
- Cleans null values
- Converts emp_length to numeric
"""

import pandas as pd
import numpy as np
import os

# Relevant columns from the Lending Club dataset
RELEVANT_COLUMNS = [
    'loan_amnt',
    'annual_inc',
    'installment',
    'dti',
    'emp_length',
    'fico_range_high',
    'fico_range_low',
    'loan_status'
]

# Valid loan statuses for binary classification
VALID_STATUSES = ['Fully Paid', 'Charged Off', 'Default']


def _convert_emp_length(series: pd.Series) -> pd.Series:
    """Convert emp_length strings (e.g. '10+ years', '< 1 year') to numeric."""
    mapping = {
        '< 1 year': 0.5,
        '1 year': 1,
        '2 years': 2,
        '3 years': 3,
        '4 years': 4,
        '5 years': 5,
        '6 years': 6,
        '7 years': 7,
        '8 years': 8,
        '9 years': 9,
        '10+ years': 10,
    }
    return series.map(mapping)


def load_dataset(filepath: str = None, sample_frac: float = None) -> pd.DataFrame:
    """
    Load and clean the Lending Club dataset.
    
    Parameters
    ----------
    filepath : str, optional
        Path to the CSV file. If None, auto-detects from project root.
    sample_frac : float, optional
        Fraction of data to sample (for faster development). None = full dataset.
        
    Returns
    -------
    pd.DataFrame
        Cleaned dataset with relevant columns.
    """
    if filepath is None:
        # Auto-detect dataset path relative to project root
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        filepath = os.path.join(
            base, 'dataset', 'accepted_2007_to_2018q4.csv',
            'accepted_2007_to_2018Q4.csv'
        )

    print(f"[DataLoader] Loading dataset from: {filepath}")
    
    # Load only relevant columns for memory efficiency
    df = pd.read_csv(filepath, usecols=RELEVANT_COLUMNS, low_memory=False)
    print(f"[DataLoader] Raw shape: {df.shape}")

    # ------------------------------------------------------------------
    # 1. Filter to valid loan statuses
    # ------------------------------------------------------------------
    df = df[df['loan_status'].isin(VALID_STATUSES)].copy()
    print(f"[DataLoader] After status filter: {df.shape}")

    # ------------------------------------------------------------------
    # 2. Convert emp_length to numeric
    # ------------------------------------------------------------------
    df['emp_length'] = _convert_emp_length(df['emp_length'])

    # ------------------------------------------------------------------
    # 3. Create a single credit_score column (avg of fico range)
    # ------------------------------------------------------------------
    df['credit_score'] = (df['fico_range_high'] + df['fico_range_low']) / 2
    df.drop(columns=['fico_range_high', 'fico_range_low'], inplace=True)

    # ------------------------------------------------------------------
    # 4. Drop rows with any remaining nulls
    # ------------------------------------------------------------------
    df.dropna(inplace=True)
    print(f"[DataLoader] After cleaning nulls: {df.shape}")

    # ------------------------------------------------------------------
    # 5. Optional sampling
    # ------------------------------------------------------------------
    if sample_frac is not None and 0 < sample_frac < 1:
        df = df.sample(frac=sample_frac, random_state=42).reset_index(drop=True)
        print(f"[DataLoader] After sampling ({sample_frac}): {df.shape}")

    print("[DataLoader] Columns:", list(df.columns))
    return df


def load_bank_policies(filepath: str = None) -> pd.DataFrame:
    """
    Load bank policies CSV.
    
    Parameters
    ----------
    filepath : str, optional
        Path to bank_policies.csv. If None, auto-detects from project root.
    
    Returns
    -------
    pd.DataFrame
        Bank policies with columns: bank, min_credit_score, min_income, max_dti, interest_rate
    """
    if filepath is None:
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        filepath = os.path.join(base, 'data', 'bank_policies.csv')
    
    df = pd.read_csv(filepath)
    print(f"[DataLoader] Loaded {len(df)} bank policies.")
    return df


# ---- Quick test when run directly ----
if __name__ == '__main__':
    df = load_dataset(sample_frac=0.01)
    print(df.head())
    print(df.dtypes)
    print("\nLoan status distribution:")
    print(df['loan_status'].value_counts())
    
    banks = load_bank_policies()
    print("\nBank Policies:")
    print(banks)

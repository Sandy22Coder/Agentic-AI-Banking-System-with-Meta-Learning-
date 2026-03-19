"""
model_training.py
=================
Train and evaluate ML models for Risk and Approval prediction.

Risk Models:
  - Logistic Regression (baseline)
  - Random Forest (final model)

Approval Model:
  - Gradient Boosting (final model)

Evaluation metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report
)


def evaluate_model(model, X_test, y_test, model_name: str = "Model") -> dict:
    """
    Evaluate a trained model and print metrics.
    
    Returns
    -------
    dict
        Dictionary of metric name → value.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1_score': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_proba),
    }
    
    print(f"\n{'='*50}")
    print(f"  {model_name} -- Evaluation Results")
    print(f"{'='*50}")
    for k, v in metrics.items():
        print(f"  {k:>12s}: {v:.4f}")
    print(f"{'='*50}")
    print(classification_report(y_test, y_pred, target_names=['Class 0', 'Class 1']))
    
    return metrics


def train_risk_models(X_train, X_test, y_train, y_test):
    """
    Train Logistic Regression and Random Forest for risk prediction.
    Returns the Random Forest as the final model.
    """
    # --- Logistic Regression (baseline) ---
    print("\n[Training] Risk Model -- Logistic Regression")
    lr = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    lr.fit(X_train, y_train)
    evaluate_model(lr, X_test, y_test, "Risk -- Logistic Regression")
    
    # --- Random Forest (final) ---
    print("\n[Training] Risk Model -- Random Forest")
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=10,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    rf_metrics = evaluate_model(rf, X_test, y_test, "Risk -- Random Forest (Final)")
    
    return rf, rf_metrics


def train_approval_model(X_train, X_test, y_train, y_test):
    """
    Train Gradient Boosting for approval prediction.
    """
    print("\n[Training] Approval Model -- Gradient Boosting")
    gb = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    )
    gb.fit(X_train, y_train)
    gb_metrics = evaluate_model(gb, X_test, y_test, "Approval -- Gradient Boosting")
    
    return gb, gb_metrics


def save_model(model, filepath: str):
    """Save a model to disk using joblib."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)
    print(f"[ModelSave] Saved to {filepath}")


def load_model(filepath: str):
    """Load a model from disk."""
    model = joblib.load(filepath)
    print(f"[ModelLoad] Loaded from {filepath}")
    return model


# ---- Quick test ----
if __name__ == '__main__':
    from data_loader import load_dataset
    from feature_engineering import create_targets, engineer_features, prepare_data
    
    # Load a small sample for testing
    df = load_dataset(sample_frac=0.05)
    df = create_targets(df)
    df = engineer_features(df)
    
    # Train risk models
    X_tr, X_te, y_tr, y_te, scaler = prepare_data(df, 'risk_target')
    risk_model, _ = train_risk_models(X_tr, X_te, y_tr, y_te)
    
    # Train approval model
    X_tr, X_te, y_tr, y_te, _ = prepare_data(df, 'approval_target')
    approval_model, _ = train_approval_model(X_tr, X_te, y_tr, y_te)

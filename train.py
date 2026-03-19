"""
train.py
========
Entry point to run the full model training pipeline.

Usage:
    python train.py
    python train.py --sample 0.1   # Use 10% of data for faster training
"""

import os
import sys
import argparse

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from data_loader import load_dataset
from feature_engineering import create_targets, engineer_features, prepare_data
from model_training import (
    train_risk_models, train_approval_model,
    save_model, evaluate_model
)


def main(sample_frac=None):
    """Run the complete training pipeline."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    # =====================================================================
    # Step 1: Load and clean dataset
    # =====================================================================
    print("\n" + "=" * 70)
    print("  STEP 1: Loading Dataset")
    print("=" * 70)
    df = load_dataset(sample_frac=sample_frac)
    
    # =====================================================================
    # Step 2: Create targets
    # =====================================================================
    print("\n" + "=" * 70)
    print("  STEP 2: Creating Targets")
    print("=" * 70)
    df = create_targets(df)
    
    # =====================================================================
    # Step 3: Engineer features
    # =====================================================================
    print("\n" + "=" * 70)
    print("  STEP 3: Feature Engineering")
    print("=" * 70)
    df = engineer_features(df)
    
    # =====================================================================
    # Step 4a: Train Risk Models
    # =====================================================================
    print("\n" + "=" * 70)
    print("  STEP 4a: Training Risk Models")
    print("=" * 70)
    X_train, X_test, y_train, y_test, scaler = prepare_data(df, 'risk_target')
    risk_model, risk_metrics = train_risk_models(X_train, X_test, y_train, y_test)
    
    # Save risk model and scaler
    save_model(risk_model, os.path.join(models_dir, 'risk_model.pkl'))
    save_model(scaler, os.path.join(models_dir, 'scaler.pkl'))
    
    # =====================================================================
    # Step 4b: Train Approval Model
    # =====================================================================
    print("\n" + "=" * 70)
    print("  STEP 4b: Training Approval Model")
    print("=" * 70)
    X_train_a, X_test_a, y_train_a, y_test_a, _ = prepare_data(df, 'approval_target')
    approval_model, approval_metrics = train_approval_model(
        X_train_a, X_test_a, y_train_a, y_test_a
    )
    
    # Save approval model
    save_model(approval_model, os.path.join(models_dir, 'approval_model.pkl'))
    
    # =====================================================================
    # Summary
    # =====================================================================
    print("\n" + "=" * 70)
    print("  TRAINING COMPLETE — SUMMARY")
    print("=" * 70)
    print(f"\n  Risk Model (Random Forest):")
    for k, v in risk_metrics.items():
        print(f"    {k:>12s}: {v:.4f}")
    
    print(f"\n  Approval Model (Gradient Boosting):")
    for k, v in approval_metrics.items():
        print(f"    {k:>12s}: {v:.4f}")
    
    print(f"\n  Models saved to: {models_dir}")
    print(f"  Files: risk_model.pkl, approval_model.pkl, scaler.pkl")
    print("=" * 70)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train loan processing models')
    parser.add_argument('--sample', type=float, default=None,
                        help='Fraction of data to sample (e.g. 0.1 for 10%%)')
    args = parser.parse_args()
    main(sample_frac=args.sample)

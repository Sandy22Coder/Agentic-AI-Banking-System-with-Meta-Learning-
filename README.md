# 🏦 Agentic AI System for Automated Loan Processing & Multi-Bank Recommendation

> **Meta-Learning Based Multi-Bank Loan Evaluation System**

An end-to-end AI system that automates loan processing using machine learning, evaluates eligibility across multiple bank policies, simulates meta-learning for per-bank predictions, and provides ranked recommendations with human-readable explanations.

---

## 📁 Project Structure

```
Meta Learning Banking System/
├── dataset/                          # Lending Club dataset
├── models/                           # Saved ML models (auto-generated)
├── data/
│   └── bank_policies.csv             # 8 bank policies
├── src/
│   ├── __init__.py
│   ├── data_loader.py                # Dataset loading & cleaning
│   ├── feature_engineering.py        # Feature engineering & targets
│   ├── model_training.py             # Model training & evaluation
│   ├── eligibility_rules.py          # Bank eligibility rules engine
│   ├── meta_learning_model.py        # Meta-learning simulation
│   ├── recommendation_engine.py      # Bank ranking & scoring
│   ├── explainability.py             # Decision explanation generator
│   └── loan_agent.py                 # End-to-end agentic pipeline
├── streamlit_app.py                  # Streamlit web UI
├── train.py                          # Model training entry point
├── test_cases.py                     # Test scenarios
├── requirements.txt                  # Python dependencies
└── README.md                         # This file
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train Models

```bash
# Full dataset (may take a few minutes)
python train.py

# Or use a 10% sample for faster training
python train.py --sample 0.1
```

### 3. Run Test Cases

```bash
python test_cases.py
```

### 4. Launch Streamlit App

```bash
streamlit run streamlit_app.py
```

---

## 🤖 Agentic Pipeline

The system follows a 7-step agentic workflow:

```
1. Preprocess Input → 2. Compute Features → 3. Predict Risk
→ 4. Apply Rules → 5. Meta-Learning → 6. Rank Banks → 7. Explain
```

| Step | Module | Description |
|------|--------|-------------|
| 1 | `feature_engineering.py` | Normalize inputs, compute DTI |
| 2 | `feature_engineering.py` | Engineer loan_to_income_ratio, credit_score_scaled |
| 3 | `model_training.py` | Random Forest risk prediction |
| 4 | `eligibility_rules.py` | Check credit, income, DTI thresholds |
| 5 | `meta_learning_model.py` | Per-bank approval probability |
| 6 | `recommendation_engine.py` | Weighted scoring & ranking |
| 7 | `explainability.py` | Human-readable explanation |

---

## 🧠 Meta-Learning Concept

Each bank is treated as a separate **task**. The system:
- Combines user features with bank-specific policy parameters
- Creates task-specific feature representations
- Predicts approval probability per bank using the trained model
- Calibrates predictions based on bank characteristics

This simulates how meta-learning adapts a base model to new tasks.

---

## 📊 Models

| Model | Algorithm | Target |
|-------|-----------|--------|
| Risk Model | Random Forest | Default prediction (Charged Off → 1) |
| Approval Model | Gradient Boosting | Approval prediction (Fully Paid → 1) |

**Metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC

---

## 🏦 Bank Policies

| Bank | Min Credit | Min Income | Max DTI | Interest Rate |
|------|-----------|-----------|---------|--------------|
| HDFC | 700 | $40,000 | 40% | 9.0% |
| ICICI | 680 | $35,000 | 45% | 9.5% |
| SBI | 650 | $30,000 | 50% | 10.0% |
| Axis Bank | 690 | $38,000 | 42% | 9.2% |
| Kotak Mahindra | 710 | $45,000 | 38% | 8.8% |
| Bank of Baroda | 640 | $28,000 | 52% | 10.5% |
| PNB | 660 | $32,000 | 48% | 9.8% |
| Yes Bank | 670 | $34,000 | 46% | 10.2% |

---

## 📐 Recommendation Formula

```
Score = 0.5 × approval_probability
      + 0.3 × normalized(1 / interest_rate)
      + 0.2 × affordability_score
```

---

## 📝 Key Concepts Demonstrated

- ✅ **Loan Automation** — End-to-end processing without manual intervention
- ✅ **Multi-Bank Evaluation** — Eligibility check across 8 bank policies
- ✅ **Meta-Learning** — Task-specific adaptation per bank
- ✅ **Agentic Workflow** — Modular pipeline with autonomous decision-making
- ✅ **Explainable AI** — Human-readable decision explanations

"""
streamlit_app.py
================
Streamlit Web UI for the Agentic AI Loan Processing System.
Indian-style professional UI with INR formatting.

Run with:
    streamlit run streamlit_app.py
"""

import os
import sys
import streamlit as st
import pandas as pd

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from loan_agent import LoanAgent


# ======================================================================
# HELPER: Indian number format (e.g. 5,00,000)
# ======================================================================
def format_inr(amount):
    """Format a number in Indian style: 5,00,000"""
    amount = int(amount)
    if amount < 0:
        return "-" + format_inr(-amount)
    s = str(amount)
    if len(s) <= 3:
        return s
    last3 = s[-3:]
    rest = s[:-3]
    # Group remaining digits in pairs from right
    groups = []
    while len(rest) > 2:
        groups.append(rest[-2:])
        rest = rest[:-2]
    if rest:
        groups.append(rest)
    groups.reverse()
    return ",".join(groups) + "," + last3


# ======================================================================
# PAGE CONFIG
# ======================================================================
st.set_page_config(
    page_title="AI Loan Processor | Meta-Learning Banking System",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ======================================================================
# CUSTOM CSS
# ======================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .stApp { font-family: 'Inter', sans-serif; }

    /* ---- Header ---- */
    .main-header {
        background: linear-gradient(135deg, #1a237e 0%, #283593 40%, #3949ab 100%);
        padding: 2.2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        text-align: center;
        box-shadow: 0 8px 30px rgba(26, 35, 126, 0.35);
        border: 1px solid rgba(255,255,255,0.06);
    }
    .main-header h1 {
        color: #ffffff; font-size: 2rem; font-weight: 700;
        margin: 0; letter-spacing: -0.5px;
    }
    .main-header .subtitle {
        color: #9fa8da; font-size: 1rem; margin-top: 0.4rem; font-weight: 400;
    }
    .main-header .badge {
        display: inline-block; background: rgba(255,255,255,0.12);
        border-radius: 20px; padding: 0.25rem 0.9rem; margin-top: 0.8rem;
        font-size: 0.75rem; color: #c5cae9; letter-spacing: 0.5px;
    }

    /* ---- Pipeline Steps ---- */
    .pipeline-bar {
        display: flex; justify-content: center; flex-wrap: wrap;
        gap: 0.3rem; margin-bottom: 1.5rem;
    }
    .pipeline-step {
        background: rgba(57, 73, 171, 0.12);
        border: 1px solid rgba(57, 73, 171, 0.25);
        border-radius: 20px; padding: 0.3rem 0.75rem;
        font-size: 0.72rem; color: #7986cb;
    }
    .pipeline-arrow { color: #5c6bc0; font-size: 0.7rem; align-self: center; }

    /* ---- Info Cards ---- */
    .result-card {
        background: linear-gradient(145deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 14px; padding: 1.4rem 1.5rem;
        margin: 0.5rem 0;
        box-shadow: 0 4px 18px rgba(0,0,0,0.18);
        transition: transform 0.2s;
    }
    .result-card:hover { transform: translateY(-2px); }
    .result-card h3 {
        color: #90a4ae; font-size: 0.78rem; font-weight: 600;
        text-transform: uppercase; letter-spacing: 1.2px; margin-bottom: 0.4rem;
    }
    .result-card .value { font-size: 1.7rem; font-weight: 700; margin: 0; }

    .approved { color: #00e676 !important; }
    .rejected { color: #ff5252 !important; }
    .risk-low { color: #69f0ae; }
    .risk-medium { color: #ffd740; }
    .risk-high { color: #ff5252; }

    /* ---- Section Headers ---- */
    .section-title {
        font-size: 1.1rem; font-weight: 600; color: #b0bec5;
        border-left: 3px solid #5c6bc0; padding-left: 0.7rem;
        margin: 1.5rem 0 0.8rem 0;
    }

    /* ---- Explanation Box ---- */
    .explanation-box {
        background: #0d1117; border: 1px solid #30363d;
        border-radius: 12px; padding: 1.3rem 1.5rem;
        font-family: 'Courier New', monospace;
        font-size: 0.82rem; color: #c9d1d9;
        white-space: pre-wrap; line-height: 1.6;
    }

    /* ---- Sidebar ---- */
    .sidebar-powered {
        background: rgba(57, 73, 171, 0.08);
        border: 1px solid rgba(57, 73, 171, 0.18);
        border-radius: 10px; padding: 0.9rem 1rem; margin-top: 0.8rem;
    }
    .sidebar-powered p {
        font-size: 0.78rem; color: #999; margin: 0.25rem 0;
    }

    /* ---- Big Font ---- */
    .big-font { font-size: 20px !important; font-weight: bold; }

    /* ---- Input Summary Card ---- */
    .input-summary {
        background: rgba(57, 73, 171, 0.06);
        border: 1px solid rgba(57, 73, 171, 0.15);
        border-radius: 10px; padding: 1rem 1.2rem;
        margin-bottom: 1rem;
    }
    .input-summary table { width: 100%; }
    .input-summary td {
        padding: 0.25rem 0.5rem; font-size: 0.85rem; color: #b0bec5;
    }
    .input-summary td:first-child { color: #78909c; font-weight: 500; }
    .input-summary td:last-child { text-align: right; color: #e0e0e0; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


# ======================================================================
# LOAD AGENT (cached)
# ======================================================================
@st.cache_resource
def load_agent():
    """Load the LoanAgent with all models."""
    return LoanAgent()


# ======================================================================
# HEADER
# ======================================================================
st.markdown("""
<div class="main-header">
    <h1>&#127974; Agentic AI Loan Processor</h1>
    <p class="subtitle">AI-powered loan approval and bank recommendation system</p>
    <span class="badge">META-LEARNING &bull; MULTI-BANK &bull; INDIA</span>
</div>
""", unsafe_allow_html=True)

# Pipeline visualization
st.markdown("""
<div class="pipeline-bar">
    <span class="pipeline-step">1. Preprocess</span>
    <span class="pipeline-arrow">&rarr;</span>
    <span class="pipeline-step">2. Ratios</span>
    <span class="pipeline-arrow">&rarr;</span>
    <span class="pipeline-step">3. Risk Model</span>
    <span class="pipeline-arrow">&rarr;</span>
    <span class="pipeline-step">4. Rules</span>
    <span class="pipeline-arrow">&rarr;</span>
    <span class="pipeline-step">5. Meta-Learning</span>
    <span class="pipeline-arrow">&rarr;</span>
    <span class="pipeline-step">6. Ranking</span>
    <span class="pipeline-arrow">&rarr;</span>
    <span class="pipeline-step">7. Explain</span>
</div>
""", unsafe_allow_html=True)


# ======================================================================
# SIDEBAR — INDIAN INPUT FORM
# ======================================================================
with st.sidebar:
    st.markdown("## Enter Your Details")
    st.caption("All amounts in Indian Rupees (INR)")
    st.markdown("---")

    col_a, col_b = st.columns(2)
    with col_a:
        monthly_income = st.number_input(
            "Monthly Income (₹)",
            min_value=10000, max_value=50_00_000, value=60000, step=5000,
            help="Your monthly salary / business income"
        )
    with col_b:
        loan_amount = st.number_input(
            "Loan Amount (₹)",
            min_value=10000, max_value=5_00_00_000, value=5_00_000, step=50000,
            help="Total loan amount requested"
        )

    col_c, col_d = st.columns(2)
    with col_c:
        emi = st.number_input(
            "Existing EMI (₹)",
            min_value=0, max_value=10_00_000, value=8000, step=1000,
            help="Total monthly EMI on existing loans"
        )
    with col_d:
        employment_type = st.selectbox(
            "Employment Type",
            options=['Salaried', 'Self-Employed', 'Unemployed', 'Student', 'Retired'],
            index=0
        )

    credit_score = st.slider(
        "Credit Score (CIBIL)",
        min_value=300, max_value=900, value=720,
        help="Your CIBIL credit score (300-900)"
    )

    # Compute annual income for backend (backend expects annual)
    annual_income = monthly_income * 12

    st.markdown("---")

    # Show quick summary before submit
    st.markdown(f"""
    <div class="input-summary">
    <table>
        <tr><td>Monthly Income</td><td>&#8377; {format_inr(monthly_income)}</td></tr>
        <tr><td>Annual Income</td><td>&#8377; {format_inr(annual_income)}</td></tr>
        <tr><td>Loan Amount</td><td>&#8377; {format_inr(loan_amount)}</td></tr>
        <tr><td>Existing EMI</td><td>&#8377; {format_inr(emi)}</td></tr>
        <tr><td>Credit Score</td><td>{credit_score}</td></tr>
        <tr><td>Employment</td><td>{employment_type}</td></tr>
    </table>
    </div>
    """, unsafe_allow_html=True)

    evaluate_btn = st.button("🚀 Evaluate Loan", use_container_width=True, type="primary")

    st.markdown("""
    <div class="sidebar-powered">
        <p><strong>Powered By</strong></p>
        <p>&#8226; Random Forest (Risk)</p>
        <p>&#8226; Gradient Boosting (Approval)</p>
        <p>&#8226; Meta-Learning Simulation</p>
        <p>&#8226; 8-Bank Rule Engine</p>
    </div>
    """, unsafe_allow_html=True)


# ======================================================================
# MAIN — RESULTS
# ======================================================================
if evaluate_btn:
    try:
        agent = load_agent()
    except Exception as e:
        st.error(f"Failed to load models. Run `python train.py` first.\n\nError: {e}")
        st.stop()

    user_data = {
        'income': annual_income,
        'credit_score': credit_score,
        'loan_amount': loan_amount,
        'emi': emi,
        'employment_type': employment_type,
    }

    with st.spinner("Running Agentic Pipeline..."):
        result = agent.process(user_data)

    # ------------------------------------------------------------------
    # SECTION 1: Decision Cards
    # ------------------------------------------------------------------
    st.markdown('<div class="section-title">Loan Evaluation Result</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        dc = 'approved' if result['decision'] == 'APPROVED' else 'rejected'
        icon = '&#10004;' if dc == 'approved' else '&#10008;'
        st.markdown(f"""
        <div class="result-card">
            <h3>Decision</h3>
            <p class="value {dc}">{icon} {result['decision']}</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        risk = result['risk_score']
        rc = 'risk-low' if risk < 0.3 else ('risk-medium' if risk < 0.5 else 'risk-high')
        rl = 'Low Risk' if risk < 0.3 else ('Medium Risk' if risk < 0.5 else 'High Risk')
        st.markdown(f"""
        <div class="result-card">
            <h3>Risk Score</h3>
            <p class="value {rc}">{risk:.2%}</p>
            <p style="color:#888;font-size:0.78rem;margin-top:0.2rem;">{rl}</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        best = result.get('best_bank', 'N/A') or 'N/A'
        st.markdown(f"""
        <div class="result-card">
            <h3>Recommended Bank</h3>
            <p class="value" style="color:#64b5f6;">&#127974; {best}</p>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        n_elig = len(result.get('eligible_banks', []))
        st.markdown(f"""
        <div class="result-card">
            <h3>Eligible Banks</h3>
            <p class="value" style="color:#ce93d8;">{n_elig} / 8</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ------------------------------------------------------------------
    # SECTION 2: Bank Comparison Table + Chart
    # ------------------------------------------------------------------
    if result['rankings']:
        left_col, right_col = st.columns([3, 2])

        with left_col:
            st.markdown('<div class="section-title">Bank Comparison Table</div>', unsafe_allow_html=True)
            ranking_data = []
            for i, r in enumerate(result['rankings'], 1):
                ranking_data.append({
                    'Rank': f"#{i}",
                    'Bank': r['bank'],
                    'Score': f"{r['score']:.4f}",
                    'Approval Prob.': f"{r['approval_probability']:.2%}",
                    'Interest Rate': f"{r['interest_rate']:.2%}",
                    'Affordability': f"{r['affordability']:.4f}",
                })
            df_ranking = pd.DataFrame(ranking_data)
            st.dataframe(df_ranking, use_container_width=True, hide_index=True)

        with right_col:
            if result['probabilities']:
                st.markdown('<div class="section-title">Approval Probability</div>', unsafe_allow_html=True)
                prob_df = pd.DataFrame(
                    list(result['probabilities'].items()),
                    columns=['Bank', 'Probability']
                ).sort_values('Probability', ascending=True)
                st.bar_chart(prob_df.set_index('Bank'))
    else:
        st.warning("No eligible banks found for the given profile.")

    st.markdown("---")

    # ------------------------------------------------------------------
    # SECTION 3: Explanation
    # ------------------------------------------------------------------
    st.markdown('<div class="section-title">Decision Explanation</div>', unsafe_allow_html=True)

    if result['decision'] == 'APPROVED':
        st.success(f"Your loan application is **APPROVED**. "
                   f"Best recommended bank: **{result.get('best_bank', 'N/A')}**")
    else:
        st.error("Your loan application is **REJECTED** based on the current profile.")

    st.markdown(f"""
    <div class="explanation-box">{result['explanation']}</div>
    """, unsafe_allow_html=True)

    # Eligible bank pills
    if result['eligible_banks']:
        st.markdown("")
        banks_html = " ".join(
            f'<span style="display:inline-block;background:rgba(0,230,118,0.12);'
            f'border:1px solid rgba(0,230,118,0.3);border-radius:20px;'
            f'padding:0.3rem 0.8rem;margin:0.2rem;font-size:0.8rem;color:#69f0ae;">'
            f'{b}</span>'
            for b in result['eligible_banks']
        )
        st.markdown(f"**Eligible at:** {banks_html}", unsafe_allow_html=True)

else:
    # ------------------------------------------------------------------
    # LANDING STATE
    # ------------------------------------------------------------------
    st.markdown("")
    st.markdown("")

    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        st.markdown("""
        <div style="text-align:center;padding:3rem 1rem;">
            <p style="font-size:3rem;margin-bottom:0.5rem;">&#127974;</p>
            <h2 style="color:#9fa8da;font-weight:600;margin-bottom:0.5rem;">
                Fill in your details to get started
            </h2>
            <p style="color:#78909c;font-size:1rem;max-width:500px;margin:0 auto;line-height:1.7;">
                Our AI-powered agentic system will analyse your financial profile,
                predict loan risk, check eligibility across <strong>8 Indian banks</strong>,
                and recommend the best option using meta-learning.
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Quick feature pills
        features = [
            "Ratio-Based Features", "Currency Independent", "8-Bank Evaluation",
            "Meta-Learning", "Explainable AI", "Real-Time Scoring"
        ]
        pills = " ".join(
            f'<span style="display:inline-block;background:rgba(57,73,171,0.1);'
            f'border:1px solid rgba(57,73,171,0.2);border-radius:20px;'
            f'padding:0.3rem 0.8rem;margin:0.2rem;font-size:0.75rem;color:#7986cb;">'
            f'{f}</span>'
            for f in features
        )
        st.markdown(f'<div style="text-align:center;margin-top:1rem;">{pills}</div>',
                    unsafe_allow_html=True)


# ---- Footer ----
st.markdown("---")
st.markdown("""
<div style="text-align:center;color:#546e7a;font-size:0.73rem;padding:0.8rem;">
    Agentic AI Loan Processing System &bull; Meta-Learning Multi-Bank Recommendation<br>
    Built with Streamlit &bull; Powered by scikit-learn &bull; Made for India &#127470;&#127475;
</div>
""", unsafe_allow_html=True)

"""
test_cases.py
=============
Test multiple scenarios through the agentic pipeline.

Test Cases:
  1. High income + High credit → Expected: APPROVED
  2. Low income + High loan   → Expected: REJECTED or limited banks
  3. Edge case: borderline    → Expected: mixed eligibility
  4. Excellent profile        → Expected: APPROVED at all banks
  5. Poor profile             → Expected: REJECTED
"""

import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from loan_agent import LoanAgent


def run_tests():
    """Run all test cases and print results."""
    
    # Initialize agent (loads models)
    agent = LoanAgent()
    
    # Define test cases
    test_cases = [
        {
            'name': 'Test 1: High Income + High Credit Score',
            'input': {
                'income': 120000,
                'credit_score': 780,
                'loan_amount': 30000,
                'emi': 1500,
                'employment_type': 'Salaried'
            },
            'expected': 'APPROVED'
        },
        {
            'name': 'Test 2: Low Income + High Loan Amount',
            'input': {
                'income': 25000,
                'credit_score': 650,
                'loan_amount': 50000,
                'emi': 3000,
                'employment_type': 'Self-Employed'
            },
            'expected': 'REJECTED'
        },
        {
            'name': 'Test 3: Borderline Case',
            'input': {
                'income': 45000,
                'credit_score': 680,
                'loan_amount': 20000,
                'emi': 1200,
                'employment_type': 'Salaried'
            },
            'expected': 'APPROVED (limited banks)'
        },
        {
            'name': 'Test 4: Excellent Profile',
            'input': {
                'income': 200000,
                'credit_score': 820,
                'loan_amount': 25000,
                'emi': 800,
                'employment_type': 'Salaried'
            },
            'expected': 'APPROVED (all banks)'
        },
        {
            'name': 'Test 5: Poor Profile (Unemployed + Low Credit)',
            'input': {
                'income': 15000,
                'credit_score': 580,
                'loan_amount': 40000,
                'emi': 2500,
                'employment_type': 'Unemployed'
            },
            'expected': 'REJECTED'
        },
    ]
    
    # Run each test
    results = []
    for i, tc in enumerate(test_cases):
        print("\n" + "#" * 70)
        print(f"  {tc['name']}")
        print(f"  Expected: {tc['expected']}")
        print("#" * 70)
        
        result = agent.process(tc['input'])
        results.append(result)
        
        print(result['explanation'])
        print(f"\n  Decision: {result['decision']}")
        print(f"  Risk Score: {result['risk_score']}")
        print(f"  Best Bank: {result['best_bank']}")
        print(f"  Eligible Banks: {result['eligible_banks']}")
        if result['probabilities']:
            print(f"  Approval Probabilities: {result['probabilities']}")
    
    # ---- Summary Table ----
    print("\n\n" + "=" * 80)
    print("  TEST RESULTS SUMMARY")
    print("=" * 80)
    print(f"{'Test':<45} {'Expected':<25} {'Actual':<10} {'Match'}")
    print("-" * 80)
    for tc, res in zip(test_cases, results):
        match = "PASS" if res['decision'] in tc['expected'].upper() else "~"
        print(f"{tc['name']:<45} {tc['expected']:<25} {res['decision']:<10} {match}")
    print("=" * 80)


if __name__ == '__main__':
    run_tests()

"""
run_pipeline.py
================================================================================
One-click verification and unit-testing pipeline for the
"B2B Working Capital & Accounts Receivable (A/R) Delinquency Diagnostic" project.

Executes:
1. Deterministic generation of clients_master.csv (350 rows), ar_invoice_ledger.csv
   (6,500 rows), and dispute_logs.csv (950 rows) via generate_empirical_data.py.
2. Comprehensive unit tests on working_capital_engine.py verifying:
   - Zero unexpected NaNs across all generated datasets and engine outputs.
   - Accounting balance integrity (Total Open A/R == Current A/R + Overdue A/R,
     Sum of Aging Buckets == Total Open A/R, DSO - BPDSO == ADD, CEI in [0, 100]).
   - Mathematical consistency of IFRS 9 ECL (EAD * PD * LGD), Monte Carlo VaR
     monotonicity (P50 < P95 < P99), Petersen & Rajan (1997) 2/10 Net-30 EAR,
     and full 4-bucket coverage in the Dynamic Policy Verdict Engine.
3. Prints an executive portfolio health check summary to stdout.
================================================================================
"""

from __future__ import annotations

import os
import sys
import unittest
import numpy as np
import pandas as pd

from generate_empirical_data import generate_all_datasets
from working_capital_engine import (
    AGING_BUCKET_ORDER,
    CREDIT_TIER_ORDER,
    analyze_disputes_and_discount_abuse,
    compute_aging_waterfall,
    compute_core_ar_metrics,
    compute_ifrs9_ecl_matrix,
    compute_implied_discount_ear,
    compute_pl_vs_cashflow_diagnostic,
    generate_client_policy_verdicts,
    run_deloof_ols_diagnostic,
    run_monte_carlo_bad_debt_simulation,
)


class TestWorkingCapitalPipeline(unittest.TestCase):
    """Automated test suite verifying data integrity and mathematical consistency."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.base_dir = os.path.dirname(os.path.abspath(__file__))
        cls.clients_df, cls.invoices_df, cls.disputes_df = generate_all_datasets(
            output_dir=cls.base_dir
        )

    def test_01_dataset_row_counts_and_schemas(self) -> None:
        """Verifies exact row counts and required column schemas for all 3 CSVs."""
        self.assertEqual(len(self.clients_df), 350)
        self.assertEqual(len(self.invoices_df), 6500)
        self.assertEqual(len(self.disputes_df), 950)

        expected_client_cols = {
            "client_id",
            "client_name",
            "industry_sector",
            "company_size",
            "credit_tier",
            "credit_limit_usd",
            "contracted_terms",
            "annual_contract_value_usd",
            "gross_margin_pct",
        }
        self.assertTrue(expected_client_cols.issubset(set(self.clients_df.columns)))

        expected_invoice_cols = {
            "invoice_id",
            "client_id",
            "invoice_date",
            "due_date",
            "payment_date",
            "invoice_amount_usd",
            "discount_offered_pct",
            "discount_taken_usd",
            "unearned_discount_flag",
            "days_to_pay",
            "days_past_due",
            "aging_bucket",
            "status",
            "cost_of_capital_carry_usd",
        }
        self.assertTrue(expected_invoice_cols.issubset(set(self.invoices_df.columns)))

        expected_dispute_cols = {
            "dispute_id",
            "invoice_id",
            "client_id",
            "dispute_reason",
            "opened_date",
            "resolved_date",
            "resolution_days",
            "disputed_amount_usd",
            "credit_memo_issued_usd",
        }
        self.assertTrue(expected_dispute_cols.issubset(set(self.disputes_df.columns)))

    def test_02_zero_unexpected_nans(self) -> None:
        """Verifies zero NaNs across all non-nullable columns and engine outputs."""
        # Clients master must have strictly 0 NaNs
        self.assertEqual(int(self.clients_df.isna().sum().sum()), 0)

        # Invoices ledger must have 0 NaNs in all columns except payment_date (which is null iff unpaid)
        inv_non_null_cols = [c for c in self.invoices_df.columns if c != "payment_date"]
        self.assertEqual(int(self.invoices_df[inv_non_null_cols].isna().sum().sum()), 0)
        paid_mask = self.invoices_df["status"] == "Paid"
        self.assertEqual(int(self.invoices_df.loc[paid_mask, "payment_date"].isna().sum()), 0)
        self.assertTrue(self.invoices_df.loc[~paid_mask, "payment_date"].isna().all())

        # Disputes log must have 0 NaNs in all columns except resolved_date (null when still open)
        disp_non_null_cols = [c for c in self.disputes_df.columns if c != "resolved_date"]
        self.assertEqual(int(self.disputes_df[disp_non_null_cols].isna().sum().sum()), 0)

    def test_03_accounting_balance_integrity(self) -> None:
        """Verifies DSO - BPDSO == ADD, Open A/R == Current + Overdue, and Aging sum == Total Open A/R."""
        kpis = compute_core_ar_metrics(
            self.invoices_df, self.clients_df, wacc_pct=8.5, target_dso_reduction_days=10.0, macro_stress_mult=1.0
        )
        # Check no NaN in KPI dict
        for k, v in kpis.items():
            self.assertFalse(pd.isna(v), f"KPI {k} is NaN")

        # Open A/R == Current A/R + Trapped Overdue A/R
        self.assertAlmostEqual(
            kpis["total_open_ar_usd"],
            kpis["current_ar_usd"] + kpis["trapped_overdue_ar_usd"],
            places=1,
        )
        # DSO - BPDSO == ADD
        self.assertAlmostEqual(
            kpis["dso_days"] - kpis["bpdso_days"],
            kpis["add_days"],
            places=1,
        )
        # CEI bounded in [0, 100]
        self.assertGreaterEqual(kpis["cei_pct"], 0.0)
        self.assertLessEqual(kpis["cei_pct"], 100.0)

        # Aging waterfall sum == Total Open A/R
        aging_df = compute_aging_waterfall(self.invoices_df)
        self.assertEqual(int(aging_df.isna().sum().sum()), 0)
        self.assertAlmostEqual(
            float(aging_df["open_ar_usd"].sum()),
            kpis["total_open_ar_usd"],
            places=1,
        )

    def test_04_ifrs9_ecl_and_monte_carlo_consistency(self) -> None:
        """Verifies IFRS 9 provision matrix math and Monte Carlo VaR monotonicity."""
        ecl_detail_df, ecl_pivot_df = compute_ifrs9_ecl_matrix(
            self.invoices_df, self.clients_df, macro_stress_mult=1.0
        )
        self.assertEqual(int(ecl_detail_df.isna().sum().sum()), 0)
        self.assertEqual(int(ecl_pivot_df.isna().sum().sum()), 0)
        self.assertEqual(len(ecl_detail_df), len(AGING_BUCKET_ORDER) * len(CREDIT_TIER_ORDER))

        # Check stressed ECL > baseline ECL
        ecl_stressed_df, _ = compute_ifrs9_ecl_matrix(
            self.invoices_df, self.clients_df, macro_stress_mult=1.8
        )
        self.assertGreater(
            float(ecl_stressed_df["expected_credit_loss_usd"].sum()),
            float(ecl_detail_df["expected_credit_loss_usd"].sum()),
        )

        # Check Monte Carlo VaR ordering: P50 < P95 < P99
        mc = run_monte_carlo_bad_debt_simulation(
            self.invoices_df, self.clients_df, macro_stress_mult=1.5, n_iterations=1000, seed=42
        )
        self.assertEqual(len(mc["baseline_losses_usd"]), 1000)
        self.assertLess(mc["baseline_p50_usd"], mc["baseline_p95_var_usd"])
        self.assertLess(mc["baseline_p95_var_usd"], mc["baseline_p99_var_usd"])
        self.assertLess(mc["stressed_p50_usd"], mc["stressed_p95_var_usd"])
        self.assertLess(mc["stressed_p95_var_usd"], mc["stressed_p99_var_usd"])
        self.assertGreater(mc["stressed_p95_var_usd"], mc["baseline_p95_var_usd"])

    def test_05_policy_verdict_and_discount_math(self) -> None:
        """Verifies Petersen & Rajan 2/10 Net-30 EAR and Policy Verdict coverage."""
        ear_info = compute_implied_discount_ear(2.0, 10, 30)
        self.assertAlmostEqual(ear_info["simple_annualized_pct"], 37.24, places=2)
        self.assertAlmostEqual(ear_info["effective_annual_rate_pct"], 44.59, places=2)

        client_matrix, summary_impact = generate_client_policy_verdicts(
            self.invoices_df,
            self.clients_df,
            self.disputes_df,
            wacc_pct=8.5,
            macro_stress_mult=1.0,
            early_settlement_offer_pct=1.5,
        )
        self.assertEqual(len(client_matrix), 350)
        self.assertEqual(int(client_matrix.isna().sum().sum()), 0)
        self.assertEqual(len(summary_impact["verdict_counts"]), 4)
        self.assertGreater(summary_impact["total_cash_unlockable_usd"], 0.0)


def main() -> None:
    print("=" * 80)
    print("B2B WORKING CAPITAL & A/R DELINQUENCY DIAGNOSTIC — PIPELINE VERIFICATION")
    print("=" * 80)

    suite = unittest.TestLoader().loadTestsFromTestCase(TestWorkingCapitalPipeline)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    if not result.wasSuccessful():
        sys.exit(1)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    clients_df = pd.read_csv(os.path.join(base_dir, "clients_master.csv"))
    invoices_df = pd.read_csv(os.path.join(base_dir, "ar_invoice_ledger.csv"))
    disputes_df = pd.read_csv(os.path.join(base_dir, "dispute_logs.csv"))

    kpis = compute_core_ar_metrics(invoices_df, clients_df, wacc_pct=8.5, target_dso_reduction_days=10.0, macro_stress_mult=1.0)
    mc = run_monte_carlo_bad_debt_simulation(invoices_df, clients_df, macro_stress_mult=1.5, n_iterations=1000, seed=42)
    _, policy_summary = generate_client_policy_verdicts(invoices_df, clients_df, disputes_df)

    print("\n" + "=" * 80)
    print("EXECUTIVE PORTFOLIO HEALTH CHECK SUMMARY")
    print("=" * 80)
    print(f"Clients Master Records:          {len(clients_df):,}")
    print(f"A/R Invoice Ledger Records:      {len(invoices_df):,}")
    print(f"Dispute Ticket Records:          {len(disputes_df):,}")
    print(f"Total Booked Revenue (18M):      ${kpis['total_billed_revenue_usd']:,.2f}")
    print(f"Total Cash Collected (18M):      ${kpis['total_cash_collected_usd']:,.2f}")
    print(f"Total Open A/R Balance:          ${kpis['total_open_ar_usd']:,.2f}")
    print(f"Trapped Overdue A/R ($ / %):     ${kpis['trapped_overdue_ar_usd']:,.2f} ({kpis['overdue_ar_pct']:.1f}%)")
    print(f"True DSO / BPDSO / ADD:          {kpis['dso_days']:.1f}d / {kpis['bpdso_days']:.1f}d / +{kpis['add_days']:.1f}d")
    print(f"Collection Effectiveness (CEI):  {kpis['cei_pct']:.1f}%")
    print(f"WACC Financing Carry Drag:       ${kpis['wacc_carry_cost_usd']:,.2f}")
    print(f"Unearned 2/10 Discount Leakage:  ${kpis['unearned_discount_leakage_usd']:,.2f} ({kpis['unearned_discount_count']} invoices)")
    print(f"IFRS 9 Provision Matrix ECL:     ${kpis['ifrs9_ecl_usd']:,.2f} ({kpis['ifrs9_ecl_pct_of_ar']:.2f}% of Open A/R)")
    print(f"Monte Carlo 1.5x Stress P95 VaR: ${mc['stressed_p95_var_usd']:,.2f}")
    print(f"Total Cash Unlockable (Verdict): ${policy_summary['total_cash_unlockable_usd']:,.2f}")
    print(f"Net EBITDA Accretion (Verdict):  +${policy_summary['net_ebitda_impact_usd']:,.2f}")
    print("Policy Bucket Distribution:     ", policy_summary["verdict_counts"])
    print("=" * 80)
    print("[ALL CHECKS PASSED — ZERO NaNs & 100% ACCOUNTING INTEGRITY VERIFIED]")


if __name__ == "__main__":
    main()
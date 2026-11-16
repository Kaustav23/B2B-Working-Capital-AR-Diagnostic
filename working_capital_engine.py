"""
working_capital_engine.py
================================================================================
Pure analytical & financial modeling backend for the
"B2B Working Capital & Accounts Receivable (A/R) Delinquency Diagnostic" engine.

Contains zero Streamlit UI dependencies so it can be unit-tested independently,
imported into batch pipelines, or wired into executive dashboards.

Key Financial & Econometric Modules:
1. Core Liquidity & A/R Metrics (CRF & Hackett Group Standard Formulas):
   - Days Sales Outstanding (DSO), Best Possible DSO (BPDSO), Average Days
     Delinquent (ADD = DSO - BPDSO), Collection Effectiveness Index (CEI),
     Overdue A/R %, Top-10 Client Concentration Risk %, Trapped Working Capital ($),
     and Cash Unlocked from Target DSO Reduction ($).
2. P&L vs. Cash Flow Illusion Diagnostic (Richards & Laughlin 1980; Deloof 2003):
   - Monthly and cumulative reconciliation of Booked Accounting Revenue & Gross
     Profit vs. Actual Cash Collected, uncollected receivable build, and WACC
     financing drag.
3. IFRS 9 / ASC 326 (CECL) Expected Credit Loss (ECL) & Monte Carlo Stress Engine:
   - Segmented provision matrix (Aging Bucket x Credit Tier) calculating
     ECL = EAD x PD x LGD, plus 1,000-iteration correlated Monte Carlo default
     simulation under baseline and recession stress multipliers (P50, P95 VaR, P99 VaR).
4. Unearned Discount & Dispute Leakage Analyzer (Petersen & Rajan 1997; EFILOS/CRF):
   - Quantifies margin erosion from 2/10 Net-30 discount abuse (44.59% EAR implied
     rate) and separates internal billing failures (Missing PO/GRN, Pricing, Tax)
     from client stall tactics.
5. Dynamic Credit & Collections Policy Verdict Engine:
   - Classifies every client account into one of four actionable policy interventions
     and simulates the net EBITDA and cash-flow impact of early-settlement offers.
================================================================================
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats

try:
    import statsmodels.api as sm  # type: ignore
except ImportError:
    sm = None

# -----------------------------------------------------------------------------
# EMPIRICAL IFRS 9 / ASC 326 (CECL) PROVISION MATRIX PARAMETERS
# Grounded in PwC IFRS 9 Practical Guide (FAQ 45.13.2) & Deloitte/IAS Plus
# ASC 326 Example [49] lifetime loss rate progressions across aging buckets.
# -----------------------------------------------------------------------------
AGING_BUCKET_ORDER: List[str] = [
    "Current",
    "1-30 DPD",
    "31-60 DPD",
    "61-90 DPD",
    "90+ DPD",
]

CREDIT_TIER_ORDER: List[str] = [
    "Tier 1 Prime",
    "Tier 2 Standard",
    "Tier 3 Watchlist",
    "Tier 4 High-Risk",
]

BASE_AGING_PD: Dict[str, float] = {
    "Current": 0.008,    # 0.8% base lifetime default probability when current
    "1-30 DPD": 0.045,   # 4.5% base PD in 1-30 DPD
    "31-60 DPD": 0.140,  # 14.0% base PD in 31-60 DPD
    "61-90 DPD": 0.320,  # 32.0% base PD in 61-90 DPD
    "90+ DPD": 0.680,    # 68.0% base PD past 90 days (Atradius/MSB 90-day cliff)
}

TIER_RISK_PARAMS: Dict[str, Dict[str, float]] = {
    "Tier 1 Prime": {"pd_mult": 0.35, "lgd": 0.45},
    "Tier 2 Standard": {"pd_mult": 0.85, "lgd": 0.60},
    "Tier 3 Watchlist": {"pd_mult": 1.35, "lgd": 0.75},
    "Tier 4 High-Risk": {"pd_mult": 1.85, "lgd": 0.88},
}


def compute_implied_discount_ear(
    discount_pct: float = 2.0,
    discount_days: int = 10,
    net_days: int = 30,
) -> Dict[str, float]:
    """
    Computes the implied annualized financing cost of two-part trade credit terms
    (e.g., 2/10 Net-30) per Petersen & Rajan (1997), Review of Financial Studies.

    Formula:
      Simple Annualized Rate = [d / (1 - d)] * [365 / (Net Days - Discount Days)]
      Effective Annual Rate (EAR) = (1 + d / (1 - d)) ** (365 / (Net Days - Discount Days)) - 1
    """
    d = max(0.0, min(discount_pct / 100.0, 0.50))
    accel_days = max(1, net_days - discount_days)
    if d == 0.0:
        return {"simple_annualized_pct": 0.0, "effective_annual_rate_pct": 0.0, "acceleration_days": float(accel_days)}
    period_rate = d / (1.0 - d)
    periods_per_year = 365.0 / float(accel_days)
    simple_rate = period_rate * periods_per_year * 100.0
    ear = ((1.0 + period_rate) ** periods_per_year - 1.0) * 100.0
    return {
        "simple_annualized_pct": round(simple_rate, 2),
        "effective_annual_rate_pct": round(ear, 2),
        "acceleration_days": float(accel_days),
    }


def compute_core_ar_metrics(
    invoices_df: pd.DataFrame,
    clients_df: pd.DataFrame,
    wacc_pct: float = 8.5,
    target_dso_reduction_days: float = 10.0,
    macro_stress_mult: float = 1.0,
) -> Dict[str, Any]:
    """
    Computes core working capital and A/R health metrics following Credit Research
    Foundation (CRF) and Hackett Group methodology.
    """
    if invoices_df.empty:
        return {
            "total_billed_revenue_usd": 0.0,
            "total_cash_collected_usd": 0.0,
            "total_open_ar_usd": 0.0,
            "current_ar_usd": 0.0,
            "trapped_overdue_ar_usd": 0.0,
            "overdue_ar_pct": 0.0,
            "ninety_plus_ar_usd": 0.0,
            "ninety_plus_ar_pct": 0.0,
            "dso_days": 0.0,
            "bpdso_days": 0.0,
            "add_days": 0.0,
            "cei_pct": 0.0,
            "top10_concentration_pct": 0.0,
            "wacc_carry_cost_usd": 0.0,
            "unearned_discount_leakage_usd": 0.0,
            "unearned_discount_count": 0,
            "written_off_usd": 0.0,
            "ifrs9_ecl_usd": 0.0,
            "ifrs9_ecl_pct_of_ar": 0.0,
            "cash_unlocked_from_dso_target_usd": 0.0,
            "annual_wacc_savings_from_dso_target_usd": 0.0,
            "period_days": 365,
            "daily_credit_sales_usd": 0.0,
        }

    df = invoices_df.copy()
    df["invoice_date_dt"] = pd.to_datetime(df["invoice_date"])
    min_dt = df["invoice_date_dt"].min()
    max_dt = df["invoice_date_dt"].max()
    period_days = max(30, int((max_dt - min_dt).days) + 1)

    wacc_rate = max(0.0, wacc_pct / 100.0)

    open_mask = df["status"].isin(["Open", "Disputed"])
    paid_mask = df["status"] == "Paid"
    wo_mask = df["status"] == "Written-Off"

    total_billed = float(df["invoice_amount_usd"].sum())
    total_collected = float((df.loc[paid_mask, "invoice_amount_usd"] - df.loc[paid_mask, "discount_taken_usd"]).sum())
    written_off_usd = float(df.loc[wo_mask, "invoice_amount_usd"].sum())

    open_df = df.loc[open_mask].copy()
    total_open_ar = float(open_df["invoice_amount_usd"].sum())
    current_ar = float(open_df.loc[open_df["days_past_due"] <= 0, "invoice_amount_usd"].sum())
    trapped_overdue_ar = float(open_df.loc[open_df["days_past_due"] > 0, "invoice_amount_usd"].sum())
    ninety_plus_ar = float(open_df.loc[open_df["days_past_due"] > 90, "invoice_amount_usd"].sum())

    overdue_ar_pct = (trapped_overdue_ar / total_open_ar * 100.0) if total_open_ar > 0 else 0.0
    ninety_plus_ar_pct = (ninety_plus_ar / total_open_ar * 100.0) if total_open_ar > 0 else 0.0

    cutoff_dt = max_dt - pd.Timedelta(days=180)
    recent_sales_df = df.loc[df["invoice_date_dt"] >= cutoff_dt]
    if len(recent_sales_df) >= 50 and recent_sales_df["invoice_amount_usd"].sum() > 0:
        window_days = max(30, int((max_dt - cutoff_dt).days))
        window_sales = float(recent_sales_df["invoice_amount_usd"].sum())
        daily_credit_sales = window_sales / float(window_days)
    else:
        daily_credit_sales = total_billed / float(period_days)

    dso_days = (total_open_ar / daily_credit_sales) if daily_credit_sales > 0 else 0.0
    bpdso_days = (current_ar / daily_credit_sales) if daily_credit_sales > 0 else 0.0
    add_days = max(0.0, dso_days - bpdso_days)

    # Collection Effectiveness Index (CEI) per Credit Research Foundation (CRF) formula
    # evaluated over a standard trailing 90-day quarterly collection cycle:
    # CEI = (Beginning AR + Quarterly Credit Sales - Ending Total AR) /
    #       (Beginning AR + Quarterly Credit Sales - Ending Current AR) * 100
    q_Sales = daily_credit_sales * 90.0
    beg_ar_est = total_open_ar * 0.92
    collectible_denominator = beg_ar_est + q_Sales - current_ar
    collected_numerator = beg_ar_est + q_Sales - total_open_ar - (written_off_usd * (90.0 / float(period_days)))
    if collectible_denominator > 0:
        cei_pct = float(np.clip((collected_numerator / collectible_denominator) * 100.0, 0.0, 100.0))
    else:
        cei_pct = 100.0

    if total_open_ar > 0:
        client_ar = open_df.groupby("client_id")["invoice_amount_usd"].sum().sort_values(ascending=False)
        top10_ar = float(client_ar.head(10).sum())
        top10_concentration_pct = (top10_ar / total_open_ar) * 100.0
    else:
        top10_concentration_pct = 0.0

    wacc_carry_cost_usd = float(
        (df["invoice_amount_usd"] * (wacc_rate / 365.0) * df["days_past_due"]).sum()
    )

    unearned_mask = df["unearned_discount_flag"] == True
    unearned_discount_leakage_usd = float(df.loc[unearned_mask, "discount_taken_usd"].sum())
    unearned_discount_count = int(unearned_mask.sum())

    ecl_table, _ = compute_ifrs9_ecl_matrix(df, clients_df, macro_stress_mult=macro_stress_mult)
    ifrs9_ecl_usd = float(ecl_table["expected_credit_loss_usd"].sum()) if not ecl_table.empty else 0.0
    ifrs9_ecl_pct_of_ar = (ifrs9_ecl_usd / total_open_ar * 100.0) if total_open_ar > 0 else 0.0

    effective_dso_drop = min(float(target_dso_reduction_days), dso_days)
    cash_unlocked_usd = daily_credit_sales * effective_dso_drop
    annual_wacc_savings_usd = cash_unlocked_usd * wacc_rate

    return {
        "total_billed_revenue_usd": round(total_billed, 2),
        "total_cash_collected_usd": round(total_collected, 2),
        "total_open_ar_usd": round(total_open_ar, 2),
        "current_ar_usd": round(current_ar, 2),
        "trapped_overdue_ar_usd": round(trapped_overdue_ar, 2),
        "overdue_ar_pct": round(overdue_ar_pct, 2),
        "ninety_plus_ar_usd": round(ninety_plus_ar, 2),
        "ninety_plus_ar_pct": round(ninety_plus_ar_pct, 2),
        "dso_days": round(dso_days, 2),
        "bpdso_days": round(bpdso_days, 2),
        "add_days": round(add_days, 2),
        "cei_pct": round(cei_pct, 2),
        "top10_concentration_pct": round(top10_concentration_pct, 2),
        "wacc_carry_cost_usd": round(wacc_carry_cost_usd, 2),
        "unearned_discount_leakage_usd": round(unearned_discount_leakage_usd, 2),
        "unearned_discount_count": unearned_discount_count,
        "written_off_usd": round(written_off_usd, 2),
        "ifrs9_ecl_usd": round(ifrs9_ecl_usd, 2),
        "ifrs9_ecl_pct_of_ar": round(ifrs9_ecl_pct_of_ar, 2),
        "cash_unlocked_from_dso_target_usd": round(cash_unlocked_usd, 2),
        "annual_wacc_savings_from_dso_target_usd": round(annual_wacc_savings_usd, 2),
        "period_days": period_days,
        "daily_credit_sales_usd": round(daily_credit_sales, 2),
    }


def compute_pl_vs_cashflow_diagnostic(
    invoices_df: pd.DataFrame,
    clients_df: pd.DataFrame,
    wacc_pct: float = 8.5,
) -> pd.DataFrame:
    """
    Builds a monthly time-series diagnostic comparing P&L Booked Revenue and
    Booked Gross Profit against Actual Cash Collected, Trapped Uncollected Revenue,
    and WACC Carry Drag (Richards & Laughlin 1980; Deloof 2003).
    """
    if invoices_df.empty:
        return pd.DataFrame(
            columns=[
                "year_month",
                "booked_revenue_usd",
                "booked_gross_profit_usd",
                "cash_collected_usd",
                "uncollected_gap_usd",
                "cumulative_booked_revenue_usd",
                "cumulative_cash_collected_usd",
                "cumulative_working_capital_gap_usd",
                "wacc_financing_drag_usd",
            ]
        )

    wacc_rate = max(0.0, wacc_pct / 100.0)
    merged = invoices_df.merge(
        clients_df[["client_id", "gross_margin_pct"]], on="client_id", how="left"
    )
    merged["gross_margin_pct"] = merged["gross_margin_pct"].fillna(35.0)
    merged["invoice_month"] = pd.to_datetime(merged["invoice_date"]).dt.to_period("M").astype(str)
    merged["booked_gp_usd"] = merged["invoice_amount_usd"] * (merged["gross_margin_pct"] / 100.0)
    merged["dynamic_wacc_drag_usd"] = (
        merged["invoice_amount_usd"] * (wacc_rate / 365.0) * merged["days_past_due"]
    )

    booked_monthly = (
        merged.groupby("invoice_month", as_index=False)
        .agg(
            booked_revenue_usd=("invoice_amount_usd", "sum"),
            booked_gross_profit_usd=("booked_gp_usd", "sum"),
            wacc_financing_drag_usd=("dynamic_wacc_drag_usd", "sum"),
        )
        .rename(columns={"invoice_month": "year_month"})
    )

    paid_df = merged.loc[merged["status"] == "Paid"].copy()
    if not paid_df.empty:
        paid_df["payment_month"] = pd.to_datetime(paid_df["payment_date"]).dt.to_period("M").astype(str)
        paid_df["net_cash_usd"] = paid_df["invoice_amount_usd"] - paid_df["discount_taken_usd"]
        cash_monthly = (
            paid_df.groupby("payment_month", as_index=False)
            .agg(cash_collected_usd=("net_cash_usd", "sum"))
            .rename(columns={"payment_month": "year_month"})
        )
    else:
        cash_monthly = pd.DataFrame(columns=["year_month", "cash_collected_usd"])

    diag = pd.merge(booked_monthly, cash_monthly, on="year_month", how="left").fillna(0.0)
    diag = diag.sort_values("year_month").reset_index(drop=True)
    diag["uncollected_gap_usd"] = diag["booked_revenue_usd"] - diag["cash_collected_usd"]
    diag["cumulative_booked_revenue_usd"] = diag["booked_revenue_usd"].cumsum()
    diag["cumulative_cash_collected_usd"] = diag["cash_collected_usd"].cumsum()
    diag["cumulative_working_capital_gap_usd"] = (
        diag["cumulative_booked_revenue_usd"] - diag["cumulative_cash_collected_usd"]
    )

    for col in [
        "booked_revenue_usd",
        "booked_gross_profit_usd",
        "cash_collected_usd",
        "uncollected_gap_usd",
        "cumulative_booked_revenue_usd",
        "cumulative_cash_collected_usd",
        "cumulative_working_capital_gap_usd",
        "wacc_financing_drag_usd",
    ]:
        diag[col] = diag[col].round(2)

    return diag


def compute_aging_waterfall(invoices_df: pd.DataFrame) -> pd.DataFrame:
    """Computes the Open A/R Aging Schedule across standard CRF / IFRS 9 buckets."""
    open_df = invoices_df.loc[invoices_df["status"].isin(["Open", "Disputed"])].copy()
    total_open = float(open_df["invoice_amount_usd"].sum()) if not open_df.empty else 0.0

    rows = []
    for bucket in AGING_BUCKET_ORDER:
        sub = open_df.loc[open_df["aging_bucket"] == bucket] if not open_df.empty else pd.DataFrame()
        amt = float(sub["invoice_amount_usd"].sum()) if not sub.empty else 0.0
        cnt = int(len(sub))
        share = (amt / total_open * 100.0) if total_open > 0 else 0.0
        rows.append(
            {
                "aging_bucket": bucket,
                "open_ar_usd": round(amt, 2),
                "invoice_count": cnt,
                "share_of_open_ar_pct": round(share, 2),
            }
        )
    return pd.DataFrame(rows)


def compute_ifrs9_ecl_matrix(
    invoices_df: pd.DataFrame,
    clients_df: pd.DataFrame,
    macro_stress_mult: float = 1.0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Computes the IFRS 9 / ASC 326 (CECL) Expected Credit Loss (ECL) Provision Matrix:
      ECL = Exposure at Default (EAD) * Probability of Default (PD) * Loss Given Default (LGD)
    Segmented by Aging Bucket (5 buckets) x Client Credit Tier (4 tiers).
    """
    open_df = invoices_df.loc[invoices_df["status"].isin(["Open", "Disputed"])].copy()
    merged = open_df.merge(
        clients_df[["client_id", "credit_tier"]], on="client_id", how="left"
    )
    merged["credit_tier"] = merged["credit_tier"].fillna("Tier 2 Standard")

    records: List[Dict[str, Any]] = []
    for bucket in AGING_BUCKET_ORDER:
        base_pd = BASE_AGING_PD[bucket]
        for tier in CREDIT_TIER_ORDER:
            t_params = TIER_RISK_PARAMS[tier]
            sub = merged.loc[
                (merged["aging_bucket"] == bucket) & (merged["credit_tier"] == tier)
            ]
            ead_usd = float(sub["invoice_amount_usd"].sum()) if not sub.empty else 0.0
            inv_cnt = int(len(sub))

            stressed_pd = float(np.clip(base_pd * t_params["pd_mult"] * macro_stress_mult, 0.001, 0.995))
            lgd = float(t_params["lgd"])
            effective_loss_rate = stressed_pd * lgd
            ecl_usd = ead_usd * effective_loss_rate

            records.append(
                {
                    "aging_bucket": bucket,
                    "credit_tier": tier,
                    "invoice_count": inv_cnt,
                    "ead_open_ar_usd": round(ead_usd, 2),
                    "pd_pct": round(stressed_pd * 100.0, 2),
                    "lgd_pct": round(lgd * 100.0, 2),
                    "effective_loss_rate_pct": round(effective_loss_rate * 100.0, 2),
                    "expected_credit_loss_usd": round(ecl_usd, 2),
                }
            )

    ecl_detail_df = pd.DataFrame(records)
    ecl_pivot_df = ecl_detail_df.pivot(
        index="aging_bucket", columns="credit_tier", values="expected_credit_loss_usd"
    ).reindex(index=AGING_BUCKET_ORDER, columns=CREDIT_TIER_ORDER).fillna(0.0)

    return ecl_detail_df, ecl_pivot_df


def run_monte_carlo_bad_debt_simulation(
    invoices_df: pd.DataFrame,
    clients_df: pd.DataFrame,
    macro_stress_mult: float = 1.0,
    n_iterations: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Runs a 1,000-iteration correlated Monte Carlo portfolio credit loss simulation
    comparing Baseline (1.0x) vs. User-Selected Macroeconomic Recession Stress (macro_stress_mult).
    """
    open_df = invoices_df.loc[invoices_df["status"].isin(["Open", "Disputed"])].copy()
    if open_df.empty:
        zeros = np.zeros(n_iterations)
        return {
            "baseline_losses_usd": zeros,
            "stressed_losses_usd": zeros,
            "baseline_p50_usd": 0.0,
            "baseline_p95_var_usd": 0.0,
            "baseline_p99_var_usd": 0.0,
            "stressed_p50_usd": 0.0,
            "stressed_p95_var_usd": 0.0,
            "stressed_p99_var_usd": 0.0,
            "unexpected_tail_loss_p95_usd": 0.0,
        }

    merged = open_df.merge(
        clients_df[["client_id", "credit_tier"]], on="client_id", how="left"
    )
    merged["credit_tier"] = merged["credit_tier"].fillna("Tier 2 Standard")

    ead_vec = merged["invoice_amount_usd"].values.astype(float)
    base_pd_vec = np.array([BASE_AGING_PD.get(b, 0.05) for b in merged["aging_bucket"]])
    tier_mult_vec = np.array([TIER_RISK_PARAMS.get(t, TIER_RISK_PARAMS["Tier 2 Standard"])["pd_mult"] for t in merged["credit_tier"]])
    lgd_vec = np.array([TIER_RISK_PARAMS.get(t, TIER_RISK_PARAMS["Tier 2 Standard"])["lgd"] for t in merged["credit_tier"]])

    p_baseline = np.clip(base_pd_vec * tier_mult_vec, 0.001, 0.995)
    p_stressed = np.clip(base_pd_vec * tier_mult_vec * macro_stress_mult, 0.001, 0.995)

    rng = np.random.default_rng(seed)
    sys_shocks_base = rng.lognormal(mean=-0.045, sigma=0.30, size=(n_iterations, 1))
    sys_shocks_stress = rng.lognormal(mean=-0.060, sigma=0.35, size=(n_iterations, 1))

    prob_matrix_base = np.clip(sys_shocks_base * p_baseline[None, :], 0.0, 0.999)
    prob_matrix_stress = np.clip(sys_shocks_stress * p_stressed[None, :], 0.0, 0.999)

    uniform_draws = rng.random(size=(n_iterations, len(ead_vec)))
    lgd_jitter = np.clip(rng.normal(loc=1.0, scale=0.08, size=(n_iterations, 1)), 0.75, 1.12)

    loss_given_def_vec = ead_vec * lgd_vec
    baseline_losses = ((uniform_draws < prob_matrix_base) * loss_given_def_vec[None, :] * lgd_jitter).sum(axis=1)
    stressed_losses = ((uniform_draws < prob_matrix_stress) * loss_given_def_vec[None, :] * lgd_jitter).sum(axis=1)

    b_p50 = float(np.percentile(baseline_losses, 50))
    b_p95 = float(np.percentile(baseline_losses, 95))
    b_p99 = float(np.percentile(baseline_losses, 99))

    s_p50 = float(np.percentile(stressed_losses, 50))
    s_p95 = float(np.percentile(stressed_losses, 95))
    s_p99 = float(np.percentile(stressed_losses, 99))

    return {
        "baseline_losses_usd": np.round(baseline_losses, 2),
        "stressed_losses_usd": np.round(stressed_losses, 2),
        "baseline_p50_usd": round(b_p50, 2),
        "baseline_p95_var_usd": round(b_p95, 2),
        "baseline_p99_var_usd": round(b_p99, 2),
        "stressed_p50_usd": round(s_p50, 2),
        "stressed_p95_var_usd": round(s_p95, 2),
        "stressed_p99_var_usd": round(s_p99, 2),
        "unexpected_tail_loss_p95_usd": round(max(0.0, s_p95 - s_p50), 2),
    }


def analyze_disputes_and_discount_abuse(
    invoices_df: pd.DataFrame,
    disputes_df: pd.DataFrame,
    clients_df: pd.DataFrame,
) -> Dict[str, pd.DataFrame]:
    """
    Analyzes dispute root causes (separating internal billing/operational failures
    from client stall tactics) and quantifies unearned early-payment discount leakage.
    """
    if disputes_df.empty or invoices_df.empty:
        dispute_summary_df = pd.DataFrame(
            columns=[
                "dispute_reason",
                "fault_category",
                "ticket_count",
                "disputed_amount_usd",
                "credit_memo_issued_usd",
                "avg_resolution_days",
                "avg_invoice_dpd",
                "share_of_disputed_dollars_pct",
            ]
        )
    else:
        d_merged = disputes_df.merge(
            invoices_df[["invoice_id", "days_past_due", "days_to_pay", "status"]],
            on="invoice_id",
            how="left",
        )
        d_merged["days_past_due"] = d_merged["days_past_due"].fillna(0.0)
        total_disp_usd = float(d_merged["disputed_amount_usd"].sum())

        fault_map = {
            "Missing PO/GRN": "Internal Billing / Portal Error",
            "Pricing Discrepancy": "Internal Billing / Portal Error",
            "Tax/Compliance Error": "Internal Billing / Portal Error",
            "Damaged Goods/SLA Breach": "Operational / Fulfillment Issue",
            "Stall Tactic / Unsubstantiated": "Client Stall Tactic (Bad-Faith Delay)",
        }
        d_merged["fault_category"] = d_merged["dispute_reason"].map(fault_map).fillna("Internal Billing / Portal Error")

        dispute_summary_df = (
            d_merged.groupby(["dispute_reason", "fault_category"], as_index=False)
            .agg(
                ticket_count=("dispute_id", "count"),
                disputed_amount_usd=("disputed_amount_usd", "sum"),
                credit_memo_issued_usd=("credit_memo_issued_usd", "sum"),
                avg_resolution_days=("resolution_days", "mean"),
                avg_invoice_dpd=("days_past_due", "mean"),
            )
            .sort_values("disputed_amount_usd", ascending=False)
            .reset_index(drop=True)
        )
        dispute_summary_df["share_of_disputed_dollars_pct"] = (
            (dispute_summary_df["disputed_amount_usd"] / total_disp_usd * 100.0)
            if total_disp_usd > 0
            else 0.0
        )
        for col in [
            "disputed_amount_usd",
            "credit_memo_issued_usd",
            "avg_resolution_days",
            "avg_invoice_dpd",
            "share_of_disputed_dollars_pct",
        ]:
            dispute_summary_df[col] = dispute_summary_df[col].round(2)

    abuse_inv = invoices_df.loc[invoices_df["unearned_discount_flag"] == True].copy()
    if abuse_inv.empty:
        abuse_client_df = pd.DataFrame(
            columns=[
                "client_id",
                "client_name",
                "industry_sector",
                "company_size",
                "credit_tier",
                "contracted_terms",
                "abused_invoice_count",
                "abused_invoice_volume_usd",
                "unearned_discount_taken_usd",
                "avg_days_to_pay_when_abused",
                "avg_days_past_discount_window",
            ]
        )
    else:
        abuse_merged = abuse_inv.merge(
            clients_df[
                [
                    "client_id",
                    "client_name",
                    "industry_sector",
                    "company_size",
                    "credit_tier",
                    "contracted_terms",
                ]
            ],
            on="client_id",
            how="left",
        )
        abuse_merged["days_past_disc_window"] = np.maximum(0, abuse_merged["days_to_pay"] - 10)
        abuse_client_df = (
            abuse_merged.groupby(
                [
                    "client_id",
                    "client_name",
                    "industry_sector",
                    "company_size",
                    "credit_tier",
                    "contracted_terms",
                ],
                as_index=False,
            )
            .agg(
                abused_invoice_count=("invoice_id", "count"),
                abused_invoice_volume_usd=("invoice_amount_usd", "sum"),
                unearned_discount_taken_usd=("discount_taken_usd", "sum"),
                avg_days_to_pay_when_abused=("days_to_pay", "mean"),
                avg_days_past_discount_window=("days_past_disc_window", "mean"),
            )
            .sort_values("unearned_discount_taken_usd", ascending=False)
            .reset_index(drop=True)
        )
        for col in [
            "abused_invoice_volume_usd",
            "unearned_discount_taken_usd",
            "avg_days_to_pay_when_abused",
            "avg_days_past_discount_window",
        ]:
            abuse_client_df[col] = abuse_client_df[col].round(2)

    return {
        "dispute_summary_df": dispute_summary_df,
        "abuse_client_df": abuse_client_df,
        "abuse_invoices_df": abuse_inv,
    }


def generate_client_policy_verdicts(
    invoices_df: pd.DataFrame,
    clients_df: pd.DataFrame,
    disputes_df: pd.DataFrame,
    wacc_pct: float = 8.5,
    macro_stress_mult: float = 1.0,
    early_settlement_offer_pct: float = 1.5,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Dynamic Credit & Collections Policy Verdict Engine.
    Classifies every client account into one of four actionable policy buckets:
      1. Immediate Credit Hold & Escalation
      2. Automated Dunning & Clawback Enforcement
      3. Dynamic Early-Settlement Incentive (e.g., 1.5/10 Net-45)
      4. VIP Frictionless Auto-Renewal
    """
    wacc_rate = max(0.0, wacc_pct / 100.0)
    offer_rate = max(0.0, early_settlement_offer_pct / 100.0)

    if invoices_df.empty or clients_df.empty:
        empty_cols = [
            "client_id",
            "client_name",
            "industry_sector",
            "company_size",
            "credit_tier",
            "contracted_terms",
            "gross_margin_pct",
            "credit_limit_usd",
            "total_open_ar_usd",
            "overdue_ar_usd",
            "credit_utilization_pct",
            "client_add_days",
            "ifrs9_ecl_usd",
            "wacc_carry_cost_usd",
            "unearned_discount_usd",
            "total_disputes",
            "stall_disputes",
            "admin_po_disputes",
            "risk_score",
            "policy_verdict",
            "recommended_action_rationale",
            "estimated_cash_unlock_usd",
            "estimated_ebitda_impact_usd",
        ]
        return pd.DataFrame(columns=empty_cols), {
            "total_cash_unlockable_usd": 0.0,
            "bad_debt_prevented_usd": 0.0,
            "discount_clawback_recovered_usd": 0.0,
            "wacc_carry_saved_usd": 0.0,
            "early_pay_discount_cost_usd": 0.0,
            "net_ebitda_impact_usd": 0.0,
            "verdict_counts": {},
        }

    df = invoices_df.copy()
    df_clients = clients_df.copy()

    tier_lookup = df_clients.set_index("client_id")["credit_tier"].to_dict()
    df["credit_tier"] = df["client_id"].map(tier_lookup).fillna("Tier 2 Standard")
    open_mask = df["status"].isin(["Open", "Disputed"])

    base_pd_vec = np.array([BASE_AGING_PD.get(b, 0.05) for b in df["aging_bucket"]])
    tier_mult_vec = np.array([TIER_RISK_PARAMS.get(t, TIER_RISK_PARAMS["Tier 2 Standard"])["pd_mult"] for t in df["credit_tier"]])
    lgd_vec = np.array([TIER_RISK_PARAMS.get(t, TIER_RISK_PARAMS["Tier 2 Standard"])["lgd"] for t in df["credit_tier"]])
    stressed_pd_vec = np.clip(base_pd_vec * tier_mult_vec * macro_stress_mult, 0.001, 0.995)

    df["inv_ecl_usd"] = np.where(
        open_mask,
        df["invoice_amount_usd"] * stressed_pd_vec * lgd_vec,
        0.0,
    )
    df["inv_open_ar_usd"] = np.where(open_mask, df["invoice_amount_usd"], 0.0)
    df["inv_overdue_ar_usd"] = np.where(open_mask & (df["days_past_due"] > 0), df["invoice_amount_usd"], 0.0)
    df["inv_90p_ar_usd"] = np.where(open_mask & (df["days_past_due"] > 90), df["invoice_amount_usd"], 0.0)
    df["inv_wacc_carry_usd"] = df["invoice_amount_usd"] * (wacc_rate / 365.0) * df["days_past_due"]
    df["inv_unearned_disc_usd"] = np.where(df["unearned_discount_flag"] == True, df["discount_taken_usd"], 0.0)

    inv_agg = (
        df.groupby("client_id", as_index=False)
        .agg(
            total_open_ar_usd=("inv_open_ar_usd", "sum"),
            overdue_ar_usd=("inv_overdue_ar_usd", "sum"),
            ninety_plus_ar_usd=("inv_90p_ar_usd", "sum"),
            client_add_days=("days_past_due", "mean"),
            ifrs9_ecl_usd=("inv_ecl_usd", "sum"),
            wacc_carry_cost_usd=("inv_wacc_carry_usd", "sum"),
            unearned_discount_usd=("inv_unearned_disc_usd", "sum"),
            unearned_discount_cnt=("unearned_discount_flag", "sum"),
            invoice_count=("invoice_id", "count"),
        )
    )

    if not disputes_df.empty:
        d_df = disputes_df.copy()
        d_df["is_stall"] = (d_df["dispute_reason"] == "Stall Tactic / Unsubstantiated").astype(int)
        d_df["is_admin_po"] = d_df["dispute_reason"].isin(
            ["Missing PO/GRN", "Pricing Discrepancy", "Tax/Compliance Error"]
        ).astype(int)
        disp_agg = (
            d_df.groupby("client_id", as_index=False)
            .agg(
                total_disputes=("dispute_id", "count"),
                stall_disputes=("is_stall", "sum"),
                admin_po_disputes=("is_admin_po", "sum"),
                disputed_amount_usd=("disputed_amount_usd", "sum"),
            )
        )
    else:
        disp_agg = pd.DataFrame(
            columns=["client_id", "total_disputes", "stall_disputes", "admin_po_disputes", "disputed_amount_usd"]
        )

    client_matrix = df_clients.merge(inv_agg, on="client_id", how="left").merge(
        disp_agg, on="client_id", how="left"
    )
    fill_zero_cols = [
        "total_open_ar_usd",
        "overdue_ar_usd",
        "ninety_plus_ar_usd",
        "client_add_days",
        "ifrs9_ecl_usd",
        "wacc_carry_cost_usd",
        "unearned_discount_usd",
        "unearned_discount_cnt",
        "invoice_count",
        "total_disputes",
        "stall_disputes",
        "admin_po_disputes",
        "disputed_amount_usd",
    ]
    for c in fill_zero_cols:
        client_matrix[c] = client_matrix[c].fillna(0.0)

    client_matrix["credit_utilization_pct"] = np.where(
        client_matrix["credit_limit_usd"] > 0,
        (client_matrix["total_open_ar_usd"] / client_matrix["credit_limit_usd"]) * 100.0,
        0.0,
    )

    verdicts: List[str] = []
    rationales: List[str] = []
    risk_scores: List[float] = []
    cash_unlocks: List[float] = []
    ebitda_impacts: List[float] = []

    bad_debt_prevented_total = 0.0
    clawback_recovered_total = 0.0
    wacc_saved_total = 0.0
    discount_cost_total = 0.0

    for _, row in client_matrix.iterrows():
        tier = row["credit_tier"]
        add = float(row["client_add_days"])
        open_ar = float(row["total_open_ar_usd"])
        overdue_ar = float(row["overdue_ar_usd"])
        ninety_ar = float(row["ninety_plus_ar_usd"])
        ecl = float(row["ifrs9_ecl_usd"])
        unearned_usd = float(row["unearned_discount_usd"])
        stall_cnt = int(row["stall_disputes"])
        admin_cnt = int(row["admin_po_disputes"])
        util_pct = float(row["credit_utilization_pct"])

        tier_base_score = {
            "Tier 1 Prime": 10.0,
            "Tier 2 Standard": 28.0,
            "Tier 3 Watchlist": 55.0,
            "Tier 4 High-Risk": 78.0,
        }.get(tier, 30.0)

        score = (
            tier_base_score
            + min(25.0, add * 0.55)
            + min(15.0, stall_cnt * 6.0)
            + (8.0 if unearned_usd > 500 else 0.0)
            + (7.0 if ninety_ar > 0 else 0.0)
        )
        score = float(np.clip(score, 2.0, 99.0))
        risk_scores.append(round(score, 1))

        if (
            (tier == "Tier 4 High-Risk" and (overdue_ar > 0 or add > 20.0))
            or ninety_ar > 15_000.0
            or (stall_cnt >= 2 and add >= 25.0)
            or ecl > 12_000.0
        ):
            verdict = "Immediate Credit Hold & Escalation"
            c_unlock = overdue_ar * 0.65
            bd_prev = ecl * 0.45
            w_save = c_unlock * (wacc_rate * (min(add, 60.0) / 365.0))
            ebitda_imp = bd_prev + w_save
            bad_debt_prevented_total += bd_prev
            wacc_saved_total += w_save
            rationale = (
                f"Freeze shipments immediately: ADD={add:.1f}d, Open Overdue=${overdue_ar:,.0f}, "
                f"IFRS 9 ECL=${ecl:,.0f}, Stall Disputes={stall_cnt}. Require 50% wire cure + executive payment plan."
            )
        elif unearned_usd > 0.0 or (admin_cnt >= 1 and overdue_ar > 0 and add >= 8.0):
            verdict = "Automated Dunning & Clawback Enforcement"
            clawback = unearned_usd * 0.80
            c_unlock = overdue_ar * 0.80 + clawback
            w_save = (overdue_ar * 0.80) * (wacc_rate * (min(add, 45.0) / 365.0))
            ebitda_imp = clawback + w_save
            clawback_recovered_total += clawback
            wacc_saved_total += w_save
            rationale = (
                f"Enforce debit-memo clawback on ${unearned_usd:,.0f} unearned 2/10 discounts and automate pre-due "
                f"PO/portal validation ({admin_cnt} billing/PO tickets; ADD={add:.1f}d)."
            )
        elif (
            tier in ["Tier 2 Standard", "Tier 3 Watchlist"]
            and (add >= 10.0 or overdue_ar > 0.0)
        ):
            verdict = "Dynamic Early-Settlement Incentive (e.g., 1.5/10 Net-45)"
            accelerated_ar = open_ar * 0.70
            c_unlock = accelerated_ar
            w_save = accelerated_ar * (wacc_rate * (28.0 / 365.0))
            bd_prev = ecl * 0.25
            disc_cost = accelerated_ar * offer_rate
            ebitda_imp = w_save + bd_prev - disc_cost
            bad_debt_prevented_total += bd_prev
            wacc_saved_total += w_save
            discount_cost_total += disc_cost
            rationale = (
                f"Liquidity-constrained {tier} account (ADD={add:.1f}d, Open A/R=${open_ar:,.0f}). "
                f"Deploy dynamic {early_settlement_offer_pct:.1f}%/10 early-pay offer to pull cash forward 28 days."
            )
        else:
            verdict = "VIP Frictionless Auto-Renewal"
            c_unlock = overdue_ar * 0.90
            w_save = c_unlock * (wacc_rate * (10.0 / 365.0))
            ebitda_imp = w_save
            wacc_saved_total += w_save
            rationale = (
                f"Healthy payment discipline (ADD={add:.1f}d, Util={util_pct:.1f}%, $0 unearned discount abuse). "
                f"Fast-track credit auto-renewal and remove collection touch friction."
            )

        verdicts.append(verdict)
        rationales.append(rationale)
        cash_unlocks.append(round(c_unlock, 2))
        ebitda_impacts.append(round(ebitda_imp, 2))

    client_matrix["risk_score"] = risk_scores
    client_matrix["policy_verdict"] = verdicts
    client_matrix["recommended_action_rationale"] = rationales
    client_matrix["estimated_cash_unlock_usd"] = cash_unlocks
    client_matrix["estimated_ebitda_impact_usd"] = ebitda_impacts

    for col in [
        "total_open_ar_usd",
        "overdue_ar_usd",
        "ninety_plus_ar_usd",
        "credit_utilization_pct",
        "client_add_days",
        "ifrs9_ecl_usd",
        "wacc_carry_cost_usd",
        "unearned_discount_usd",
        "disputed_amount_usd",
    ]:
        client_matrix[col] = client_matrix[col].round(2)

    client_matrix = client_matrix.sort_values(
        ["risk_score", "total_open_ar_usd"], ascending=[False, False]
    ).reset_index(drop=True)

    total_cash_unlockable = float(client_matrix["estimated_cash_unlock_usd"].sum())
    net_ebitda_impact = float(
        bad_debt_prevented_total + clawback_recovered_total + wacc_saved_total - discount_cost_total
    )
    verdict_counts = client_matrix["policy_verdict"].value_counts().to_dict()

    summary_impact = {
        "total_cash_unlockable_usd": round(total_cash_unlockable, 2),
        "bad_debt_prevented_usd": round(bad_debt_prevented_total, 2),
        "discount_clawback_recovered_usd": round(clawback_recovered_total, 2),
        "wacc_carry_saved_usd": round(wacc_saved_total, 2),
        "early_pay_discount_cost_usd": round(discount_cost_total, 2),
        "net_ebitda_impact_usd": round(net_ebitda_impact, 2),
        "verdict_counts": verdict_counts,
    }

    return client_matrix, summary_impact


def run_deloof_ols_diagnostic(client_matrix: pd.DataFrame) -> Dict[str, float]:
    """
    Runs a cross-sectional OLS regression testing the Deloof (2003) / Allianz Trade (2024)
    relationship between payment delinquency (ADD) and effective net margin erosion.
    """
    if len(client_matrix) < 10:
        return {"slope_margin_erosion_per_10d_add": 0.0, "r_squared": 0.0, "p_value": 1.0}

    df = client_matrix.copy()
    acv = np.maximum(df["annual_contract_value_usd"].values, 1.0)
    effective_drag_pct = (
        (df["wacc_carry_cost_usd"] + df["ifrs9_ecl_usd"] + df["unearned_discount_usd"]) / acv
    ) * 100.0
    x_vals = df["client_add_days"].values.astype(float)
    y_vals = effective_drag_pct.values.astype(float)

    if sm is not None:
        X = sm.add_constant(x_vals)
        model = sm.OLS(y_vals, X).fit()
        slope_10d = float(model.params[1] * 10.0) if len(model.params) > 1 else 0.0
        r_sq = float(model.rsquared)
        p_val = float(model.pvalues[1]) if len(model.pvalues) > 1 else 1.0
    else:
        res = stats.linregress(x_vals, y_vals)
        slope_10d = float(res.slope * 10.0)
        r_sq = float(res.rvalue ** 2)
        p_val = float(res.pvalue)

    return {
        "slope_margin_erosion_per_10d_add": round(slope_10d, 3),
        "r_squared": round(r_sq, 3),
        "p_value": float(p_val),
    }
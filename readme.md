# B2B Working Capital & Accounts Receivable (A/R) Delinquency Diagnostic

**Principal Financial Risk & Working Capital Architecture | Executive Streamlit Decision Engine**  
*Author: Kaustav Goswami ([GitHub: Kaustav23](https://www.google.com/search?q=https://github.com/Kaustav23))*

---

## 1. Executive Problem Statement: The "Profit vs. Cash" Illusion

B2B enterprises frequently report healthy accounting profit (**EBITDA** and **Net Income** on the accrual P&L) while experiencing severe liquidity crunches on the balance sheet. This disconnect occurs because accrual accounting books revenue the moment an invoice is issued, whereas actual cash Remains trapped when enterprise and mid-market buyers:
1. **Stretch Contractual Payment Terms:** Unilaterally extending `Net-30` invoices to `Net-75+`, effectively using the supplier as an interest-free bank ([Petersen & Rajan, 1997](https://www.researchgate.net/publication/5216727_Trade_Credit_Theories_and_Evidence); [Allianz Trade, 2024](https://www.allianz-trade.com/en_global/news-insights/economic-insights/corporate-financing-dso-wcr-2024.html)).
2. **Freeze Cash in Billing Disputes:** Blocking payment on entire invoices over administrative PO mismatches (`Missing PO/GRN`), pricing discrepancies, or bad-faith **Stall Tactics** ([EFILOS / CRF DSO Playbook, 2026](https://www.efilostech.com/academy/reducing-dso-playbook)).
3. **Abuse Early-Payment Discounts:** Deducting `2/10 Net-30` early-settlement discounts (which carry an implied **44.59% Effective Annual Rate**) while still paying on Day 35 to Day 60+.

According to [The Hackett Group 2025 U.S. Working Capital Survey](https://www.thehackettgroup.com/2025-working-capital-survey-payables-rebound-receivables-inventory-lag/), **$1.7 trillion** is trapped in excess working capital across the top 1,000 U.S. public nonfinancial companies (**$600 billion** specifically in Accounts Receivable driven by an **18-day DSO gap** between top-quartile and median firms), while the [2025 Atradius Payment Practices Barometer](https://group.atradius.com/knowledge-and-research/reports/b2b-payment-practices-trends-in-north-america-2025) documents that **40%–43% of B2B credit invoices** across North America are paid past due and **4%–6% of long-overdue invoices** are written off as uncollectible bad debt.

---

## 2. System Architecture (7-File Modular Design)

```text
+-----------------------------------------------------------------------------------+
|                   B2B WORKING CAPITAL & A/R DIAGNOSTIC PIPELINE                   |
+-----------------------------------------------------------------------------------+
                                         |
         +-------------------------------+-------------------------------+
         v                                                               v
+----------------------------------+           +------------------------------------+
|   generate_empirical_data.py     |           |   METHODOLOGY_AND_CITATIONS.md     |
|  Deterministic (Seed = 42)       |           |  - CRF / Hackett KPI Equations     |
|  Parameterized via Atradius,     |           |  - IFRS 9 / ASC 326 PD x LGD Grids |
|  Allianz Trade, CRF & IFRS 9     |           |  - Petersen & Rajan (1997) 2/10    |
+----------------------------------+           +------------------------------------+
         |
         | Generates Relational CSVs:
         |  1. clients_master.csv (350 B2B Accounts)
         |  2. ar_invoice_ledger.csv (6,500 Invoices, 18 Months)
         |  3. dispute_logs.csv (950 Linked Dispute Tickets)
         v
+-----------------------------------------------------------------------------------+
|                            working_capital_engine.py                              |
|  Pure Analytical & Econometric Backend (Zero UI Code):                            |
|  - Core Liquidity Engine: DSO, BPDSO, ADD, CEI, WACC Carry Drag, Cash Unlock      |
|  - P&L vs. Cash Flow Diagnostic: Monthly Booked Revenue/GP vs. Cash Collected     |
|  - IFRS 9 / ASC 326 Provision Matrix & 1,000-Trial Correlated Monte Carlo VaR     |
|  - Dispute Bottleneck & Unearned Discount Abuse Leakage Analyzer                  |
|  - Dynamic Credit & Collections Policy Verdict Engine (4 Action Buckets)          |
+-----------------------------------------------------------------------------------+
         |                                                               |
         v                                                               v
+----------------------------------+           +------------------------------------+
|        run_pipeline.py           |           |              app.py                |
|  One-Click Verification Suite:   |           |  5-Tab Executive Streamlit UI:     |
|  - Generates all 3 CSV datasets  |           |  - Real-time What-If Policy Sliders|
|  - Verifies zero NaNs & schema   |           |  - Plain-English KPI Subtitles     |
|  - Audits accounting identities  |           |  - Interactive Plotly Visuals      |
|  - Prints Portfolio Health Check |           |  - CSV Action Plan Export          |
+----------------------------------+           +------------------------------------+
```

---

## 3. Repository File Inventory

| File Name | Role & Description |
| :--- | :--- |
| `requirements.txt` | Pinned, compatible versions of `streamlit`, `pandas`, `numpy`, `scipy`, `plotly`, and `statsmodels`. |
| `generate_empirical_data.py` | Deterministic (`np.random.seed(42)`) relational data generator producing `clients_master.csv` (350 clients), `ar_invoice_ledger.csv` (6,500 invoices), and `dispute_logs.csv` (950 disputes). |
| `working_capital_engine.py` | Pure Python financial modeling engine computing DSO, BPDSO, ADD, CEI, WACC drag, IFRS 9 ECL provision matrices, 1,000-trial Monte Carlo VaR, and account-level Policy Verdicts. |
| `app.py` | 5-tab Executive Streamlit Dashboard featuring plain-English KPI translations under every metric and real-time What-If simulation sliders. |
| `README.md` | Complete project documentation, architecture diagram, KPI glossary, and setup guide. |
| `METHODOLOGY_AND_CITATIONS.md` | Mathematical derivations, IFRS 9 / ASC 326 PD×LGD matrices, and annotated bibliography with primary source links. |
| `run_pipeline.py` | Automated verification and unit-testing suite validating data generation, zero-NaN integrity, and mathematical consistency. |

---

## 4. Plain-English KPI Glossary

| Metric | Technical Formula | Plain-English Translation |
| :--- | :--- | :--- |
| **DSO (Days Sales Outstanding)** | $(\text{Ending Total A/R} / \text{Credit Sales}) \times \text{Days}$ | The average number of days it takes to collect cash after issuing an invoice. |
| **BPDSO (Best Possible DSO)** | $(\text{Ending Current A/R} / \text{Credit Sales}) \times \text{Days}$ | The ideal collection speed if every single customer paid on the exact day their invoice was due. |
| **ADD (Average Days Delinquent)** | $\text{DSO} - \text{BPDSO}$ | How many days late customers are paying on average beyond their agreed deadline. |
| **CEI (Collection Effectiveness Index)** | $\frac{\text{Beg A/R} + \text{Sales} - \text{End Total A/R}}{\text{Beg A/R} + \text{Sales} - \text{End Current A/R}} \times 100$ | Out of every $100 that became due for collection, how many dollars the collections team actually brought in the door. |
| **IFRS 9 / CECL Expected Credit Loss (ECL)** | $\sum \text{EAD}_{b,c} \times \text{PD}_{b,c} \times \text{LGD}_c$ | The forward-looking dollar reserve we must set aside today for unpaid invoices statistically likely to default. |
| **Unearned Discount Abuse** | $\sum \text{Discount Taken USD}\; (\text{where Paid Day} > 10)$ | Profit margin quietly lost when customers take a 2% early-payment discount despite paying weeks late. |

---

## 5. Quickstart & Execution Instructions

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate empirical datasets and run the automated unit test & health check suite
python3 run_pipeline.py

# 3. Launch the interactive Streamlit executive dashboard
streamlit run app.py
```

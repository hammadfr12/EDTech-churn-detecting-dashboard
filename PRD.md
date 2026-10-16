# Payment Growth Analytics Platform

**Product Requirements Document • Version 1.0 • Portfolio / MVP Specification**

Owner: Hammad Khan | Primary BI: Tableau | Analytics Stack: SQL + Python + Tableau

- **Document version:** 1.0
- **Owner:** Hammad Khan
- **Status:** Portfolio / MVP specification
- **Primary users:** Product Managers, Account Managers, TAMs, Growth/Business teams, Data/Business Analysts
- **Primary BI tool:** Tableau
- **Analytics stack:** SQL + Python + Tableau
- **Reference/local stack:** DuckDB or PostgreSQL for local execution; Trino-compatible SQL; Pandas/NumPy/PyArrow; FastAPI; Streamlit optional
- **Data:** Synthetic payment/merchant data for portfolio development unless connected to an authorized real data source

---

## 1. Executive Summary

The Payment Growth Analytics Platform is a self-service analytics product designed to help payment businesses understand merchant performance, benchmark merchants against relevant peer groups, investigate payment-success-rate movements, and identify actionable growth opportunities.

The core product loop is:

**Monitor → Benchmark → Diagnose → Identify Opportunity → Recommend Action → Measure Impact**

The MVP combines:
1. A standardized analytical data model.
2. SQL transformation and KPI logic.
3. Python data-quality, analytics, and opportunity-detection pipelines.
4. Tableau dashboards and a Tableau-ready Hyper extract.
5. A local API/service layer for reusable metrics.
6. Optional Streamlit UI for a fully runnable local demonstration.
7. Tests, documentation, synthetic data generation, and reproducible setup.

---

## 2. Problem Statement

Payment performance varies across merchants, industries, payment methods, issuers/banks, devices, platforms, geographies, and time periods.

Stakeholders frequently need answers such as:

- Why did a merchant's success rate decline?
- Which payment method contributed most to the decline?
- Is the merchant underperforming its industry?
- Which banks/issuers are contributing to the movement?
- Is the issue concentrated on Android, iOS, or Web?
- Which merchants represent the largest conversion opportunity?
- Did a product/process change improve payment performance?

Without a centralized analytics product, these questions can require repeated SQL work and manual reporting.

**Product problem:** Create a trusted self-service platform that allows a stakeholder to move from a KPI movement to a data-backed explanation and potential action without requiring a new ad-hoc analysis for every question.

---

## 3. Product Vision

Build a payment intelligence layer that transforms raw transaction data into standardized, explainable and actionable business insights.

**Vision statement:** Give payment teams one place to understand what changed, why it changed, how the merchant compares with peers, and where the next growth opportunity exists.

---

## 4. Goals and Non-Goals

### 4.1 Goals
- **G1 — Standardize payment KPIs:** One definition for SR, TPV, transaction count, AOV, checkout conversion, share, failure rate and period-over-period changes.
- **G2 — Enable self-service analysis:** Filter merchants, industries, payment methods, dates and other dimensions without writing SQL.
- **G3 — Enable benchmarking:** Compare merchant performance against industry/peer benchmarks using documented methodology.
- **G4 — Enable RCA:** Drill-down from an aggregate KPI to contributing dimensions.
- **G5 — Surface opportunities:** Detect material negative movements and benchmark gaps; turn them into prioritized opportunity cards.
- **G6 — Make insights reproducible:** Every metric generated from documented SQL/Python logic.
- **G7 — Create a portfolio-grade implementation:** Runnable locally with synthetic data; production-style engineering practices.

### 4.2 Non-goals for MVP
- Processing live payment transactions.
- Handling real customer PII.
- Making financial decisions or approving/rejecting transactions.
- Building a production payment gateway.
- Automatically changing merchant configuration.
- Claiming business impact from synthetic data.

---

## 5. Target Users

- **Product Manager:** product-level trends, impact of changes, industry comparisons, failure-pattern analysis.
- **Account Manager / TAM:** merchant-level health, benchmarking, RCA, actionable talking points.
- **Growth / Business Team:** segment opportunities, industry gaps, merchant prioritization.
- **Analyst:** trusted metric definitions, reusable datasets, drill-downs, data-quality checks.

---

## 6. Product Principles

1. Single source of truth for KPI definitions.
2. Explainability — every insight shows underlying drivers.
3. Drill-down first — aggregate metrics traceable to dimensions.
4. No magic numbers — metrics come from data.
5. Benchmark carefully — appropriate peer groups and minimum sample sizes.
6. Actionable outputs — insights lead to a possible next step.
7. Data quality before visualization.
8. Synthetic data is explicitly labeled.

---

## 7. Core Product Workflow

User selects merchant/date → Performance Overview → (Trend, Industry Benchmark, Payment Method, Bank/Issuer, Device/Platform, Error Code) → RCA → Opportunity Engine → Recommended Investigation/Action → Measure performance after action.

---

## 8. Core KPIs

- **Success Rate (SR):** `authorized transactions / eligible transactions * 100`. Successful = `authorized_at IS NOT NULL`.
- **Transaction Count:** distinct transaction IDs in the population.
- **TPV:** total payment value of eligible transactions.
- **AOV:** `TPV / successful transaction count` (NULL on divide-by-zero).
- **Checkout Conversion:** `successful payment sessions / checkout sessions started * 100` (from checkout event table, NOT derived from transactions).
- **Payment Method Share:** `method TPV / total TPV * 100`.
- **Failure Rate:** `failed eligible transactions / eligible transactions * 100`.
- **Percentage-point change (rates):** `current rate - previous rate`.
- **Percentage change (absolute metrics):** `(current - previous) / previous * 100` (NULL where previous is zero).

---

## 9. Dimensions

Merchant, Merchant segment, Industry, MCC, Date/Week/Month, Payment method, Payment type, Card type, Bank/issuer, Device, Platform/OS, Geography, Error code, Transaction status, Account Manager/owner.

---

## 10. Data Model

- **Fact `payment_transactions`:** transaction_id, merchant_id, event_timestamp, amount, currency, payment_method, payment_type, card_type, issuer_bank, industry, mcc, device, platform, country, status, error_code, authorized_at, checkout_session_id.
- **Dimension `merchants`:** merchant_id, merchant_name, industry, mcc, merchant_segment, account_manager, country, onboarded_at.
- **Dimension `payment_methods`:** payment_method, payment_type, card_type.
- **Fact `checkout_events`:** event_id, checkout_session_id, merchant_id, event_timestamp, event_type, device, platform, payment_method. Event types: checkout_started, payment_attempted, payment_success, payment_failed.
- **Dimension `error_codes`:** error_code, error_category, description, owner_team.
- **Dimension `calendar`:** date, week_start, month_start, quarter, year.

---

## 11. Data Generation

Synthetic generator simulates: 50–200 merchants, multiple industries, MCCs, payment methods, banks/issuers, devices, platforms, transaction amounts, success/fail, realistic error-code distributions, seasonal/weekly patterns, merchant-specific performance, industry baselines, controlled anomalies, merchants with SR deterioration, benchmark gaps, and checkout funnel events. Known ground-truth scenarios enable automated test validation.

---

## 12. Benchmarking Methodology

Default benchmark = `industry + country + payment method + selected time window`. Requirements: minimum sample size, minimum merchant count in peer group, exclude the selected merchant, weighted aggregate SR, display sample size and benchmark definition. Example: Merchant SR 92.1% vs industry 95.0%, gap -2.9 pp, 37 peers, 1.8M transactions.

---

## 13. RCA Methodology

Compare current vs previous period; calculate contribution by: payment method, payment type/card type, issuer/bank, device, platform, industry/MCC, error category, geography. For each: current metric, previous metric, delta, volume, contribution estimate, sample-size guardrails. Distinguish observed movement / contributing segment / correlation / potential root cause. Do not claim causality.

---

## 14. Opportunity Engine

Rules:
- **Rule A — SR deterioration:** SR falls ≥ 1.5 pp and sample exceeds minimum threshold.
- **Rule B — Benchmark gap:** merchant SR ≥ 2 pp below benchmark.
- **Rule C — Payment-method opportunity:** a method with meaningful volume underperforms overall SR or benchmark.
- **Rule D — Error spike:** an error category increases materially vs previous period.
- **Rule E — Concentrated degradation:** a device/platform/bank segment explains a meaningful share of total SR deterioration.

Every opportunity contains: opportunity_id, merchant_id, severity, type, metric, current_value, previous_value, benchmark_value, gap, affected_volume, primary_dimension, supporting_evidence, recommended_next_step, created_at.

---

## 15. Opportunity Prioritization

Transparent, rule-based: **Critical** (large absolute impact + high confidence), **Attention** (material movement + sufficient evidence), **Opportunity** (benchmark gap + meaningful volume). Rules exposed in UI.

---

## 16. Dashboard / Tableau Requirements

Five dashboards: (1) Executive Overview, (2) Merchant Performance, (3) Industry Benchmarking, (4) RCA Explorer, (5) Opportunity Center. Filters: Date, Merchant, Industry, MCC, Payment method, Geography, Device, Platform.

---

## 17. Tableau Technical Requirements

Generate Hyper extract via Tableau Hyper API; documented data source schema; calculated fields; parameters for date range; filters; dashboard actions; tooltips with metric definitions; sample-size warnings.

---

## 18. SQL Requirements

SQL models: stg_transactions, stg_merchants, stg_checkout_events, dim_calendar, fct_payment_performance, fct_checkout_conversion, merchant_daily_metrics, merchant_weekly_metrics, industry_benchmarks, payment_method_metrics, rca_metrics, opportunity_candidates. Trino-compatible style; DuckDB for local execution with documented adaptations.

---

## 19. Python Requirements

Python responsibilities: synthetic data generation, data validation, ETL orchestration, statistical calculations, RCA, opportunity detection, Hyper extract creation, API layer, automated tests.

---

## 20. API Requirements

FastAPI endpoints: `/health`, `/metrics/overview`, `/metrics/merchant/{id}`, `/metrics/industry/{industry}`, `/metrics/trend`, `/metrics/payment-method`, `/metrics/rca`, `/benchmarks/merchant/{id}`, `/opportunities`, `/opportunities/{id}`. Common query params: start_date, end_date, merchant_id, industry, mcc, payment_method, device, platform, country.

---

## 21. Optional Streamlit Application

Mirrors Tableau IA: Overview, Merchant Performance, Benchmarking, RCA, Opportunities. Runnable product demo.

---

## 22. Data Quality Requirements

Automated checks: duplicate transaction IDs, null merchant IDs, invalid amounts, invalid dates, unknown payment methods, unknown industries, missing benchmark groups, zero denominators, impossible authorization timestamps, checkout events without valid merchants, orphan references. Data-quality report with totals/valid/rejected/duplicate/nulls/invalid.

---

## 23. Testing Requirements

Unit tests (SR, AOV, share, pp/percentage delta, benchmark, RCA contribution, opportunity rules). Data tests (uniqueness, referential integrity, accepted values, non-negative amounts, date validity). Integration test (full pipeline). Regression test (fixed synthetic scenario with known answer).

---

## 24. Observability

Log: pipeline start/end, records processed/rejected, query failures, opportunity counts, API errors. Pipeline run summary JSON.

---

## 25. Security / Privacy

Synthetic data only, no secrets committed, `.env.example`, no real merchant/customer data, `.gitignore`, production security considerations documented.

---

## 26–28. Repository Structure, Milestones, Acceptance Criteria

See `CLAUDE_CODE_BUILD_PROMPT.md` for the exact structure and `BUILD_REPORT.md` for acceptance sign-off.

---

## 29. Demo Scenario

**Merchant:** Nova Fashion • **Industry:** E-commerce • **Period:** Current week.

- SR: 91.8%, Previous SR: 95.0%, Change: -3.2 pp
- Industry SR: 95.4%, Benchmark gap: -3.6 pp
- RCA: Debit Card major contributor; Android major contributor; Issuer X elevated; Error Category Y elevated
- Opportunity: merchant benchmark gap + SR deterioration
- Next step: Investigate issuer/error-category performance for Android debit-card segment

All values derived from synthetic data.

---

## 30–33. Future Roadmap, Risks, Success Metrics, Final Product Definition

**Core promise:** From "What changed?" to "Why did it change?" to "What should we investigate next?"

See full PRD text in the original `.docx`; this markdown is a faithful summary used as the source of truth during implementation. Where this summary and `CLAUDE_CODE_BUILD_PROMPT.md` conflict, the more specific requirement wins and the decision is documented.

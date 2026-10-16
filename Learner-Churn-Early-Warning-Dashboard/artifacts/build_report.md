# Build report — Learner Churn Early-Warning Dashboard

Generated 2026-10-12 20:48 UTC by `python -m src.pipeline`. Every figure below is read from a generated artifact. Anything not computed is marked **not computed** rather than estimated.

## 1. Dataset

- **Source**: https://analyse.kmi.open.ac.uk/open_dataset (OULAD, public research data)
- **Raw directory**: `data/raw`
- **Learner-course records**: 32,593
- **Distinct learners**: 28,785 (pseudonymous IDs)
- **Course presentations**: 22 across 7 modules
- **Duplicate learner-course rows**: 0

### Outcome distribution

| Outcome | Learners | Share |
|---|---:|---:|
| Pass | 12,361 | 37.9% |
| Withdrawn | 10,156 | 31.2% |
| Fail | 7,052 | 21.6% |
| Distinction | 3,024 | 9.3% |
| **Total** | **32,593** | **100.0%** |

### Assessment calendar

- 195 of 206 assessments carry a real due date (94.7%).
- Lateness is computed only on that subset. No due date is ever imputed.

### Raw tables

| File | Rows | Missing required columns |
|---|---:|---|
| `courses.csv` | 22 | none |
| `assessments.csv` | 206 | none |
| `vle.csv` | 6,364 | none |
| `studentInfo.csv` | 32,593 | none |
| `studentRegistration.csv` | 32,593 | none |
| `studentAssessment.csv` | 173,912 | none |
| `studentVle.csv` | 10,655,280 | none |

### Data warnings

- Only 94.7% of assessments carry a due date. Lateness is computed on that subset only and is reported as unavailable elsewhere; no due dates are imputed.

## 2. Feature layer

- **Week mapping**: week = floor(day / 7) + 1 for day >= 0; day 0 is week 1. Negative days are pre-course activity, reported separately and excluded from the weekly series.
- **Cutoff weeks built**: [3, 4]
- **Default cutoff week**: 4

### Snapshot shape

| Cutoff week | Rows | Columns |
|---|---:|---:|
| 3 | 32,593 | 54 |
| 4 | 32,593 | 56 |

### Join-grain checks

| Relation | Rows | Duplicate key rows | Key |
|---|---:|---:|---|
| `stg_student_info` | 32,593 | 0 | `code_module, code_presentation, id_student` |
| `learner_week_matrix` | 28,303 | 0 | `code_module, code_presentation, id_student` |
| `cohort_outcomes` | 32,593 | 0 | `code_module, code_presentation, id_student` |
| `weekly_cohort_activity` | 550 | 0 | `code_module, code_presentation, week` |

### Measured activity decline

Activity decline is **not** withdrawal. Denominator is the previous week's active learners across all cohorts.

| Week | Active (prev) | Active (now) | Net change | Drop-off |
|---:|---:|---:|---:|---:|
| 11 | 17,559 | 14,544 | 3,015 | 17.2% |
| 25 | 14,912 | 13,628 | 1,284 | 8.6% |
| 9 | 19,888 | 18,390 | 1,498 | 7.5% |
| 4 | 23,822 | 22,135 | 1,687 | 7.1% |
| 12 | 14,544 | 13,660 | 884 | 6.1% |

## 3. Model

### Target definition

- **Primary - Withdrawn vs Pass/Distinction**
- Positive = Withdrawn. Negative = Pass or Distinction. Fail rows are excluded from the modelling cohort entirely.
- Positive: `Withdrawn`
- Negative: `Pass, Distinction`
- Excluded: `Fail`

### Cutoff and cohort

- **Prediction cutoff**: end of week 4 (last day = day 27)
- **Modelling cohort**: 25,541 of 32,593 records
- **Excluded by target**: Fail = 7,052
- **Features**: 47

### Split strategy

- GroupShuffleSplit on (code_module, code_presentation, id_student); the holdout is split again 50/50 into validation and test.

| Split | Learners | Positive | Prevalence | Majority baseline |
|---|---:|---:|---:|---:|
| Train | 19,155 | 7,636 | 0.3986 | 0.6014 |
| Validation | 3,193 | 1,253 | 0.3924 | 0.6076 |
| Test | 3,193 | 1,267 | 0.3968 | 0.6032 |

### Operating threshold

- **Threshold**: 0.7337
- **Rule**: `max_f1_under_capacity` — Highest F1 among thresholds flagging at most 25% of the cohort on the validation split.
- **Chosen on**: validation split

### Held-out performance (test split)

| Metric | Value |
|---|---:|
| ROC-AUC | 0.8850 |
| PR-AUC | 0.8789 |
| Brier score | 0.1252 |
| Precision | 0.9395 |
| Recall | 0.6133 |
| F1 | 0.7421 |

### Confusion matrix (test split)

| | Predicted negative | Predicted positive |
|---|---:|---:|
| **Actual negative** | 1,876 | 50 |
| **Actual positive** | 490 | 777 |

**Group-aware cross-validation** (training partition, 5-fold): PR-AUC **0.871605** ± 0.005467

### Top feature indicators

| Feature | Permutation importance (mean) | Std dev |
|---|---:|---:|
| `overdue_count` | +0.04737 | 0.00331 |
| `code_presentation` | +0.03213 | 0.00166 |
| `days_since_last_activity` | +0.03032 | 0.00369 |
| `module_presentation_length` | +0.02642 | 0.00199 |
| `code_module` | +0.02631 | 0.00257 |
| `on_time_submissions` | +0.01826 | 0.00201 |
| `mean_score` | +0.01336 | 0.00246 |
| `weight_due_by_cutoff` | +0.00916 | 0.00115 |
| `clicks_per_active_day` | +0.00832 | 0.00165 |
| `max_score` | +0.00747 | 0.00193 |
| `mean_lateness_days` | +0.00742 | 0.00172 |
| `first_score` | +0.00708 | 0.00207 |
| `min_score` | +0.00705 | 0.00204 |
| `latest_score` | +0.00701 | 0.00206 |
| `has_no_score_by_cutoff` | +0.00696 | 0.00199 |

Permutation importance on the validation split, measured as the drop in PR-AUC when a feature is shuffled. A ranking of predictive contribution, **not** a causal explanation.

### Limitations

- Retrospective research dataset; not a live operational feed.
- Outcomes are historical. Nothing here establishes causation.
- The primary target excludes Fail; see the sensitivity target.
- Fairness-sensitive attributes (region, imd_band, age_band, gender, disability) are available to the model but must not drive intervention decisions on their own.
- Assessment lateness is only computed for the subset of assessments that carry a real due date; it is never imputed.

> This model is trained on the anonymised, retrospective Open University Learning Analytics Dataset. It is a research and demonstration artifact. It is NOT validated for real student-support decisions, it does not establish learner intent or ability, and no learner should be contacted or judged on the basis of its score without human review.

## 4. Reproducing this report

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m src.pipeline --download
pytest
streamlit run app/Home.py
```

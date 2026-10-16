# Data dictionary

Every feature the model can see, where it comes from, how the cutoff applies,
how missing values are handled, and its leakage risk.

**Grain** — one row per learner-course presentation
(`code_module`, `code_presentation`, `id_student`). Verified unique by
`tests/test_join_grain.py`.

**Cutoff** — the default prediction cutoff is the end of course-relative week 4,
i.e. day 27 inclusive. Placeholders `{{CUTOFF_WEEK}}` / `{{CUTOFF_DAY}}` in the
SQL are filled from `src/config.py`.

---

## Source tables

| File | Grain | Rows (OULAD) | Role |
|---|---|---:|---|
| `studentInfo.csv` | learner-course | 32,593 | Attributes, enrolment, **`final_result` (the label)** |
| `studentRegistration.csv` | learner-course | 32,593 | Registration / unregistration day offsets |
| `studentVle.csv` | learner-site-day | 10,655,280 | Clickstream |
| `vle.csv` | site | 6,364 | Activity catalogue |
| `studentAssessment.csv` | learner-assessment | 173,912 | Submissions and scores |
| `assessments.csv` | assessment | 206 | Assessment calendar, **due dates** |
| `courses.csv` | module-presentation | 22 | Presentation length in days |

---

## Engagement features

| Feature | Type | Definition | Source | Cutoff rule | Missing handling | Leakage risk |
|---|---|---|---|---|---|---|
| `clicks_w1` … `clicks_w4` | int | Sum of `sum_click` in that week | `studentVle.date`, `sum_click` | Only weeks ≤ cutoff are materialised | No row = **genuine 0**, `COALESCE`d | None |
| `active_days_w1` … `active_days_w4` | int | Distinct days with any click in that week | `studentVle.date` | as above | No row = **genuine 0** | None |
| `total_clicks_to_cutoff` | int | Sum of `clicks_w1..w4` | derived | Weeks ≤ cutoff only | 0 | None |
| `total_active_days_to_cutoff` | int | Sum of `active_days_w1..w4` | derived | Weeks ≤ cutoff only | 0 | None |
| `clicks_per_active_day` | float | `total_clicks / total_active_days` | derived | Weeks ≤ cutoff only | NULL when 0 active days | None |
| `pre_course_clicks` | int | Clicks on **negative** day offsets | `studentVle.date < 0` | Pre-course by definition | 0 | None — happens before the course |
| `pre_course_active_days` | int | Distinct negative-day activity days | as above | Pre-course | 0 | None |
| `active_weeks` | int | Weeks 1..8 with any activity | `learner_week_matrix` | Materialised to week 8; **snapshot uses ≤ cutoff** | 0 | **Review** — see note |

> **Note on `active_weeks`.** The wide matrix materialises weeks 1–8, but the
> feature matrix for a week-4 cutoff only selects `clicks_w1..w4` and
> `active_days_w1..w4`. `active_weeks` as written in the matrix counts weeks up
> to 8; the snapshot re-derives engagement from the cutoff-scoped columns. The
> leakage test covers the click/active-day columns that the model actually uses.

---

## Trend features

All trend features are computed **only from weeks at or before the cutoff**.

| Feature | Type | Definition | Cutoff rule | Missing handling | Leakage risk |
|---|---|---|---|---|---|
| `click_delta` | int | `clicks_w4 − clicks_w3` | Both weeks ≤ cutoff | 0 | None |
| `active_days_delta` | int | `active_days_w4 − active_days_w3` | Both weeks ≤ cutoff | 0 | None |
| `click_slope` | float | Closed-form OLS slope of clicks against week index over weeks 1..cutoff | Weeks ≤ cutoff only | 0 | None |
| `days_since_last_activity` | int | `cutoff_day − last active day within the window` | `learner_week_engagement` filtered to `week ≤ cutoff` | 0 when never active | **Guarded** — the underlying `last_active_day_in_window` is computed inside the window only |

A negative `click_slope` means engagement is decaying across the observed window.

---

## Assessment features

The distinction that matters: **what was due** (a calendar join, so an
unsubmitted assessment still counts) versus **what the learner did**.

| Feature | Type | Definition | Source | Cutoff rule | Missing handling | Leakage risk |
|---|---|---|---|---|---|---|
| `assessments_due_by_cutoff` | int | Calendar assessments with `due_day ≤ cutoff` | `assessments` | Due on/before cutoff | 0 | None |
| `assessments_due_with_date` | int | Of those, ones with a real due date | `assessments.date` not `?` | as above | 0 | None |
| `weight_due_by_cutoff` | float | Sum of assessment weights due | `assessments.weight` | as above | NULL | None |
| `submitted_by_cutoff` | int | Submissions with `submitted_day ≤ cutoff` | `studentAssessment` | Submitted on/before cutoff | 0 | None |
| `banked_submissions` | int | Of those, `is_banked = 1` | `studentAssessment` | as above | 0 | None |
| `submission_rate` | float | `submitted_by_cutoff / assessments_due_by_cutoff` | derived | as above | NULL when nothing was due | None |
| `overdue_count` | int | `max(due − submitted, 0)` | derived | as above | NULL when nothing was due | None |
| `mean_score` | float | Mean score of submissions by cutoff | `studentAssessment.score` | Submitted on/before cutoff | **NULL, never 0** | None |
| `min_score` / `max_score` | int | Score extremes by cutoff | as above | as above | NULL | None |
| `first_score` / `latest_score` | int | Score of earliest / latest submission | window function on `submitted_day` | as above | NULL | None |
| `score_delta` | int | `latest_score − first_score` | derived | as above | NULL unless ≥2 scored submissions | None |
| `mean_lateness_days` | float | Mean `submitted_day − due_day` | `studentAssessment` + `assessments.date` | as above | NULL when no dated assessment | **Guarded** |
| `max_lateness_days` | int | Maximum lateness | as above | as above | NULL | Guarded |
| `late_submissions` | int | Submissions with `lateness_days > 0` | as above | as above | NULL | Guarded |
| `on_time_submissions` | int | Submissions with `lateness_days ≤ 0` | as above | as above | NULL | Guarded |
| `lateness_unavailable` | bool | True when no lateness could be computed | derived | as above | — | None |

**Lateness is never fabricated.** `lateness_days` is populated only where
`assessments.date` is a real day offset. 11 of 206 assessments (mostly Exams)
record `?`; those submissions count towards participation but never towards a
lateness statistic.

---

## Missingness flags

These exist so the model can tell "no score observed" apart from "score of zero"
after imputation.

| Feature | Type | Meaning |
|---|---|---|
| `has_no_score_by_cutoff` | bool | `mean_score IS NULL` |
| `has_no_submission_by_cutoff` | bool | No submissions at all by the cutoff |
| `lateness_unavailable` | bool | No lateness could be computed |

---

## Context features

| Feature | Type | Source | Leakage risk |
|---|---|---|---|
| `num_of_prev_attempts` | int | `studentInfo` | None — known at enrolment |
| `studied_credits` | int | `studentInfo` | None |
| `module_presentation_length` | int | `courses` | None — a property of the presentation |
| `code_module` | cat | `studentInfo` | None |
| `code_presentation` | cat | `studentInfo` | None |
| `gender` | cat | `studentInfo` | **Fairness-sensitive** |
| `region` | cat | `studentInfo` | **Fairness-sensitive** |
| `highest_education` | cat | `studentInfo` | Fairness-sensitive |
| `imd_band` | cat | `studentInfo` | **Fairness-sensitive** (3.4% null) |
| `age_band` | cat | `studentInfo` | **Fairness-sensitive** |
| `disability` | cat | `studentInfo` | **Fairness-sensitive** |

Fairness-sensitive attributes are available to the model because they are
genuinely present at prediction time. They are flagged here and in the model
card, and the dashboard does not use them to justify an intervention.

---

## Forbidden features

Deny-listed in `src/model/features.py`. A column on this list raises
`LeakageError` if it reaches the feature matrix.

| Column | Reason |
|---|---|
| `final_result` | The label itself |
| `target`, `is_positive`, `is_negative` | Derived from the label |
| `date_unregistration` | Unregistration timing is a post-hoc outcome signal |
| `date_registration` | Pre-course enrolment artifact |
| `last_active_week`, `total_clicks` | Whole-course aggregates, not cutoff-safe |
| `last_active_day_in_window`, `last_active_week_in_window` | Raw day/week offsets; `days_since_last_activity` supersedes them |
| `last_submission_day`, `first_submission_day` | Raw offsets; superseded by submission features |
| `mean_assessment_weight` | Constant per module-presentation; no signal |

---

## Preprocessing

| Step | Numeric | Categorical |
|---|---|---|
| Imputation | Median + explicit missing indicator | Constant `"Unknown"` |
| Scaling | `StandardScaler` | — |
| Encoding | — | `OneHotEncoder(handle_unknown="ignore")` |

Fitted inside the sklearn pipeline, so it only ever sees training folds.

---

## Cohort metric definitions

| Metric | Definition | Guard |
|---|---|---|
| Learners enrolled | Distinct learner-course presentations in the cohort | Stated inclusion rules |
| Withdrawal rate | Withdrawn / enrolled | Never combined with failure |
| Completion rate | (Pass + Distinction) / enrolled | Fail and Withdrawn shown separately |
| Weekly active rate | Learners with ≥1 interaction in the week / enrolled | Weeks with no data are zero, not missing |
| Week-over-week drop-off | (active prev − active now) / active prev | Rows with `prev = 0` excluded; bounded to the cohort's last active week |
| Risk precision | TP / flagged | At the chosen threshold |
| Risk recall | TP / actual positives | Paired with precision and the confusion matrix |
| ROC-AUC / PR-AUC | Ranking quality on held-out data | PR-AUC is the informative one at 39.7% prevalence |
| Intervention coverage | Queue items with an owner / queue items | A **process** metric, not evidence of retention impact |

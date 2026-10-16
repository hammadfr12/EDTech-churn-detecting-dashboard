# Model card — Learner Churn Early-Warning baseline

## Overview

| | |
|---|---|
| **Model** | L2 logistic regression, `class_weight="balanced"`, `max_iter=2000` |
| **Task** | Binary classification: will this learner-course presentation end in *Withdrawn*? |
| **Owner** | Portfolio / demonstration project |
| **Version** | 1.0 |
| **Trained on** | OULAD (anonymised, retrospective) |
| **Artifact** | `artifacts/churn_model.joblib` |
| **Metrics** | `artifacts/model_metrics.json` |

> **Warning.** This model is trained on anonymised, retrospective research data.
> It is a demonstration artifact. It is **not validated for real
> student-support decisions**, it does not establish learner intent or ability,
> and no learner should be contacted or judged on the basis of its score without
> human review.

---

## Intended use

**Intended**
- Demonstrate a leakage-safe early-warning pipeline on public education data.
- Rank a cohort so a limited number of human reviewers can prioritise.
- Support cohort-level analysis of where activity declines.

**Not intended**
- Automated or autonomous contact with learners.
- Any punitive action (withdrawal of access, fees, escalation) based on a score.
- Judging an individual learner's effort, ability or intent.
- Any deployment on live personal data without re-validation, DPIA and consent
  review.

---

## Training data

| | |
|---|---|
| Source | Open University Learning Analytics Dataset (Kuzilek, Hlosta & Zdrahal) |
| Snapshot rows | 32,593 learner-course presentations |
| Modelling cohort | 25,541 (7,052 `Fail` rows excluded by the primary target) |
| Distinct learners | 28,785 pseudonymous IDs |
| Presentations | 22 across 7 modules |
| Features | 47 |

### Outcome distribution (all records)

| Outcome | Count | Share |
|---|---:|---:|
| Pass | 12,361 | 37.9% |
| Withdrawn | 10,156 | 31.2% |
| Fail | 7,052 | 21.6% |
| Distinction | 3,024 | 9.3% |

### Target definition

**Primary** (`primary`)

- Positive: `Withdrawn`
- Negative: `Pass`, `Distinction`
- **Excluded: `Fail`**

Fail is excluded deliberately. Failing and withdrawing are different behaviours;
folding them together would make the headline number uninterpretable. Excluding
them also removes 21.6% of records from the modelling cohort, which is a real
cost and is reported rather than hidden.

**Sensitivity** (`fail_as_churn`)

- Positive: `Withdrawn`, `Fail`
- Negative: `Pass`, `Distinction`

Provided so the reader can see how much the result depends on the primary
choice. Run with `python -m src.model.train --target fail_as_churn`.

---

## Prediction cutoff

End of **course-relative week 4** (day 27 inclusive), with week 3 built for
comparison.

```
week = floor(day / 7) + 1     for day >= 0
```

Day 0 belongs to week 1. Negative offsets are pre-course activity, excluded from
the weekly series and captured separately as `pre_course_clicks`.

---

## Split strategy

`GroupShuffleSplit` on `(code_module, code_presentation, id_student)`. The group
is a **learning instance**, so the same instance can never appear in both train
and test. The holdout is split again 50/50 into validation and test.

| Split | Learners | Positive | Prevalence | Majority baseline |
|---|---:|---:|---:|---:|
| Train | 19,155 | 7,636 | 0.3986 | 0.6014 |
| Validation | 3,193 | 1,253 | 0.3924 | 0.6076 |
| Test | 3,193 | 1,267 | 0.3968 | 0.6032 |

Preprocessing (imputation, scaling, one-hot encoding) is fitted inside the
pipeline and therefore sees training folds only.

---

## Operating threshold

**0.7337**, chosen on the **validation split** by an explicit rule:

> Among thresholds flagging at most **25%** of the cohort — the mentor-capacity
> constraint — take the one with the highest F1.

F1 rather than accuracy because with a 39.7% positive rate, accuracy rewards
predicting the majority class. The capacity constraint is the operational story:
a queue a team can actually work through.

The test set was then scored **once**, at this frozen threshold.

---

## Held-out performance

Test split, n = 3,193, prevalence 0.3968, threshold 0.733714.

| Metric | Value |
|---|---:|
| ROC-AUC | 0.884970 |
| PR-AUC | 0.878859 |
| Brier score | 0.125215 |
| Precision | 0.939541 |
| Recall | 0.613260 |
| F1 | 0.742120 |

### Confusion matrix (test)

| | Predicted negative | Predicted positive |
|---|---:|---:|
| **Actual negative** | 1,876 | 50 |
| **Actual positive** | 490 | 777 |

490 withdrawn learners were missed. 50 learners were flagged who did not
withdraw. Both are real costs, and the threshold is how the trade-off is set.

### Baselines and comparison

| Model | PR-AUC |
|---|---:|
| Majority-class baseline | 0.603 (accuracy — predicting the common class) |
| **Logistic regression, held-out test** | **0.8789** |
| Logistic regression, 5-fold grouped CV on the train partition | 0.8716 ± 0.0055 |
| Shallow decision tree (`max_depth=4`), validation | 0.8089 |

The logistic baseline is the shipped model. The tree is *worse* on validation
PR-AUC (0.809 vs the logistic model's validation PR-AUC), so there is no
accuracy-for-explainability trade-off to make here: the explainable model is
also the better-ranking one.

---

## Top feature indicators

Permutation importance on the validation split, measured as the drop in PR-AUC
when a feature is shuffled.

| Feature | Importance (mean) |
|---|---:|
| `overdue_count` | +0.0474 |
| `code_presentation` | +0.0321 |
| `days_since_last_activity` | +0.0303 |
| `module_presentation_length` | +0.0264 |
| `code_module` | +0.0263 |
| `on_time_submissions` | +0.0183 |
| `mean_score` | +0.0134 |
| `weight_due_by_cutoff` | +0.0092 |

Permutation importance is stochastic, so these values move by roughly ±0.004
between runs. The ranking is stable; the third decimal place is not.

**This is a ranking of predictive contribution in historical data, not a causal
explanation.** `code_module` and `code_presentation` ranking highly is a warning
as much as a finding: the model is partly learning which courses and which
intakes have historically had higher withdrawal rates. That is legitimate for
triage and dangerous for individual judgement, and it is a reason the queue
requires human review rather than being treated as a decision.

---

## Leakage controls

| Control | Implementation |
|---|---|
| Feature allow-list | `src/model/features.py`; undeclared columns raise `LeakageError` |
| Deny-list with reasons | `FORBIDDEN_FEATURES` in the same module |
| Cutoff-scoped SQL | Per-week columns filtered to `week <= cutoff` in `sql/04_learner_week_features.sql` |
| Grouped split | `GroupShuffleSplit` on the learning instance |
| Train-only preprocessing | Fitted inside the sklearn `Pipeline` |
| Threshold on validation | Test scored once, at a frozen threshold |
| Injection test | `tests/test_leakage.py` adds post-cutoff clicks and submissions and asserts the week-4 feature vector is unchanged |

Never used as a feature: `final_result`, post-cutoff clicks, post-cutoff scores,
`date_unregistration`, `date_registration`, whole-course aggregates.

---

## Fairness considerations

Fairness-sensitive attributes — `gender`, `region`, `imd_band`, `age_band`,
`highest_education`, `disability` — are available to the model because they are
genuinely present at prediction time.

- They are **not** used to justify an intervention anywhere in the UI.
- Group-level performance has **not** been formally audited in this build; doing
  so properly needs sample-size-aware evaluation per group and is listed as
  future work rather than claimed.
- A deployment would need an explicit decision about whether to include them at
  all, plus monitoring for disparate impact.

---

## Limitations

1. **Retrospective research data.** OULAD is not a live operational feed.
   External validity to a real LMS is unproven.
2. **Correlation, not causation.** Nothing here explains *why* a learner
   withdraws.
3. **The primary target excludes Fail.** 21.6% of records are outside the
   modelling cohort. The sensitivity target exists so the cost is visible.
4. **Course identity is predictive.** `code_module` and `code_presentation`
   carry signal that is about cohorts, not individuals.
5. **Assessment lateness is partial.** Computable for 195 of 206 assessments;
   never imputed for the rest.
6. **Recall is 0.61 at the chosen threshold.** Under the 25% queue cap, roughly
   four in ten withdrawn learners are not flagged. Raising recall means a larger
   queue.
7. **Single dataset, single institution.** No cross-institution validation.
8. **No fairness audit.** See above.
9. **No intervention impact measured.** The workflow demonstrates process
   design. No time saving or retention uplift is claimed, because none was
   measured.

---

## Ethical use

- Risk scores are **rank-orderings of observed behaviour**, not judgements.
- Human review is required before any action.
- Only pseudonymous dataset identifiers are used. No contact details are
  collected, stored or required.
- Do not present feature contributions to a learner as reasons.
- Do not describe a fall in activity as withdrawal. The dashboard keeps the two
  separate throughout.

---

## Reproducing these numbers

```bash
python -m src.pipeline --download
```

or, with the data already in `data/raw/`:

```bash
python -m src.features.build --data-dir data/raw
python -m src.model.train --cutoff-week 4
```

Every figure in this card is written to `artifacts/model_metrics.json` at
training time. Nothing is transcribed by hand.

---

## Attribution

Kuzilek, J., Hlosta, M., & Zdrahal, Z. (2017). *Open University Learning
Analytics Dataset*. Scientific Data, 4, 170171.
<https://analyse.kmi.open.ac.uk/open_dataset>

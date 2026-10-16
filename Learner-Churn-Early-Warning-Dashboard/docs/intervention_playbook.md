# Intervention playbook

How the at-risk queue is meant to be worked. This describes a **process**.
It is not evidence that any action improves retention, and nothing in this
document should be presented as a measured outcome.

---

## What this is, and what it is not

**It is** a designed workflow showing how a weekly risk queue would route a
reviewed learner to a human reviewer with a supportive action and a due date.

**It is not** an outreach system. No message is sent, no learner is contacted,
and no time saving has been measured. Claims of retention uplift would require a
controlled or credible quasi-experimental evaluation, which has not been run.

---

## Weekly cycle

### 1. Refresh the queue

```bash
python -m src.pipeline
streamlit run app/Home.py
```

Open **Risk queue**, set the cohort filters, and set the prediction week.

### 2. Review before acting

The queue is a triage list, not a decision. Before assigning anyone, check:

- **Data quality.** Does the learner actually have activity data, or is this an
  artifact of a missing feed?
- **Context.** A learner with low clicks in week 4 may be on a legitimate study
  break, studying another module, or simply working offline.
- **The indicators.** They describe association in historical data. They are not
  reasons.

### 3. Assign an owner

In **Interventions → Assign a learner**, pick from the highest-risk entries and
record:

| Field | Guidance |
|---|---|
| Owner | A mentor or reviewer handle. No personal contact data. |
| Action | One of the four supportive actions below. |
| Due date | When the review should be complete. |
| Status | Start at `Reviewed` or `Assigned`. |
| Note | One line. No contact details, no sensitive personal data. |

### 4. Work the record

Move the status through the lifecycle as the review progresses. Every update is
timestamped for audit.

### 5. Review coverage

**Intervention coverage** = queue items with an owner ÷ queue items. This is a
**process** metric: it tells you whether learners were reviewed, not whether
anything improved.

---

## Suggested actions

Deliberately supportive. Punitive options are not offered by the application.

| Action | Use when | What it means |
|---|---|---|
| **Supportive check-in** | Engagement has fallen sharply and there is no obvious cause | A human offers to talk. Not a warning. |
| **Clarify assessment requirements** | `overdue_count` is high but the learner is still active | The learner may have misunderstood what was due, or how to submit it. |
| **Share relevant learning resource** | Scores are low but participation is steady | Point to material for the specific topic, not a generic nudge. |
| **Invite to office hours** | A single steep drop after a difficult assessment | Open, optional, no penalty for not attending. |

**Never** use a score to justify a punitive action. The model is a rank-ordering
of historical behaviour, and it is wrong often enough — 490 missed withdrawn
learners and 50 false flags in the held-out test set — that treating it as a
verdict would harm learners.

---

## Status lifecycle

```
New ──▶ Reviewed ──▶ Assigned ──▶ Contact attempted ──┬──▶ Resolved
                                                      ├──▶ No response
                                                      └──▶ Not appropriate
```

| Status | Meaning |
|---|---|
| `New` | Created, not yet looked at |
| `Reviewed` | A human has examined the case and its context |
| `Assigned` | An owner has taken responsibility |
| `Contact attempted` | A supportive approach was made (simulated in this demo) |
| `Resolved` | The review concluded with an action taken |
| `No response` | Contact was attempted and nothing came back |
| `Not appropriate` | Review concluded no action was warranted — a **good** outcome |

`Not appropriate` matters. A playbook without it pressures reviewers to act on
every flag, which is how false positives turn into harm.

---

## A worked example

> The queue flags learner `123456` in `BBB 2013J` at score 0.91.
>
> **Check the data.** 340 clicks by week 4, 22 active days, no missing feed.
> **Check the context.** Mean score 71, but `overdue_count` is 2 and
> `days_since_last_activity` is 12.
>
> **Reading.** Active and performing, then stopped submitting and went quiet
> about ten days ago. That pattern — engaged, then a sharp stop — is the one
> worth a human look.
>
> **Action.** Assign to `mentor-b`, action *Supportive check-in*, due in three
> days, status `Assigned`. Note: "Active through wk3, no submission since."
>
> **Outcome.** Review finds the learner had a personal emergency. Status →
> `Resolved`. No punitive step was ever considered, which is the point.

---

## Guardrails

- **Human review is mandatory.** No score triggers an automatic action.
- **Only pseudonymous IDs.** The database stores `id_student` from OULAD and
  nothing else identifying.
- **Minimal notes.** One line of context. No contact details, no health or
  personal circumstances.
- **No punitive routing.** The action vocabulary contains no warning, penalty or
  escalation option.
- **Process metrics only.** Coverage is reported. Retention impact is not,
  because it was not measured.
- **Say what the score is.** If a learner asks, the honest answer is: "a model
  flagged a change in your activity pattern, and a person looked at it."

---

## Evaluating this properly

If this were deployed, the honest evaluation would be:

1. A **randomised or stepped-wedge** design, or a defensible quasi-experimental
   comparison, with the outcome pre-registered.
2. **Process metrics first** (coverage, time-to-review, action mix) because they
   are measurable immediately and do not require causal inference.
3. **Outcome metrics later**, with a stated counterfactual.
4. **Harm monitoring**, especially for false positives: did flagged learners who
   did not need help experience worse outcomes for having been flagged?

None of that has been done here. Do not imply otherwise.

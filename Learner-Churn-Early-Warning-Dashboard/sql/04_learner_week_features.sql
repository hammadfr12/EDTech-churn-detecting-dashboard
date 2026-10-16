-- 04_learner_week_features.sql
-- ---------------------------------------------------------------------------
-- The cutoff-safe feature snapshot. One row per learner-course presentation.
--
-- LEAKAGE CONTRACT
--   Every column produced here is derived from records observed on or before
--   {{CUTOFF_DAY}} (the last day of cutoff week {{CUTOFF_WEEK}}). The
--   following are never used as features:
--       * final_result (the label)
--       * clicks or assessment submissions after the cutoff
--       * date_unregistration
--       * assessment outcomes posted after the cutoff
--   tests/test_leakage.py proves this by injecting post-cutoff activity and
--   asserting the week-{{CUTOFF_WEEK}} feature vector is unchanged.
--
-- TEMPLATE PLACEHOLDERS (filled by src/features/build.py)
--   {{CUTOFF_WEEK}}       integer cutoff week, e.g. 4
--   {{CUTOFF_DAY}}        last day of that week, inclusive (week*7 - 1)
--   {{WEEK_COLS}}         "w.clicks_w1, w.clicks_w2, ..." up to the cutoff
--   {{TOTAL_CLICKS}}      SQL expression summing clicks over the window
--   {{TOTAL_ACTIVE_DAYS}} SQL expression summing active days over the window
--   {{CLICKS_PER_DAY}}    SQL expression for clicks per active day
--   {{CLICK_DELTA}}       SQL expression for last-week minus previous-week
--   {{ACTIVE_DELTA}}      SQL expression for the same on active days
--   {{CLICK_SLOPE}}       SQL expression for the OLS slope of clicks
-- ---------------------------------------------------------------------------

CREATE OR REPLACE VIEW learner_feature_snapshot AS
WITH base AS (
    SELECT
        i.code_module,
        i.code_presentation,
        i.id_student,
        i.final_result,
        i.gender,
        i.region,
        i.highest_education,
        i.imd_band,
        i.age_band,
        i.num_of_prev_attempts,
        i.studied_credits,
        i.disability,
        c.module_presentation_length
    FROM stg_student_info AS i
    LEFT JOIN stg_courses AS c
      ON c.code_module = i.code_module
     AND c.code_presentation = i.code_presentation
),
week_inputs AS (
    SELECT * FROM learner_week_matrix
),
-- Recency measured strictly inside the observation window.
recency AS (
    SELECT
        code_module,
        code_presentation,
        id_student,
        MAX(last_active_day) AS last_active_day_in_window,
        MAX(week)            AS last_active_week_in_window
    FROM learner_week_engagement
    WHERE week <= {{CUTOFF_WEEK}}
    GROUP BY 1, 2, 3
),
joined AS (
    SELECT
        b.code_module,
        b.code_presentation,
        b.id_student,
        b.final_result,
        b.gender,
        b.region,
        b.highest_education,
        b.imd_band,
        b.age_band,
        b.num_of_prev_attempts,
        b.studied_credits,
        b.disability,
        b.module_presentation_length,
        {{WEEK_COLS}},
        COALESCE(p.pre_course_clicks, 0)      AS pre_course_clicks,
        COALESCE(p.pre_course_active_days, 0) AS pre_course_active_days,
        r.last_active_day_in_window,
        r.last_active_week_in_window,
        a.assessments_due_by_cutoff,
        a.assessments_due_with_date,
        a.weight_due_by_cutoff,
        a.submitted_by_cutoff,
        a.banked_submissions,
        a.mean_score,
        a.min_score,
        a.max_score,
        a.mean_assessment_weight,
        a.lateness_observations,
        a.mean_lateness_days,
        a.max_lateness_days,
        a.late_submissions,
        a.on_time_submissions,
        a.last_submission_day,
        a.first_submission_day,
        a.first_score,
        a.latest_score,
        a.score_delta,
        a.overdue_count
    FROM base AS b
    LEFT JOIN week_inputs AS w USING (code_module, code_presentation, id_student)
    LEFT JOIN recency     AS r USING (code_module, code_presentation, id_student)
    LEFT JOIN learner_precourse_engagement AS p USING (code_module, code_presentation, id_student)
    LEFT JOIN learner_assessment_features  AS a USING (code_module, code_presentation, id_student)
)
SELECT
    *,
    -- ------------------------------------------------------------------
    -- Engagement aggregates over the window
    -- ------------------------------------------------------------------
    ({{TOTAL_CLICKS}})       AS total_clicks_to_cutoff,
    ({{TOTAL_ACTIVE_DAYS}})  AS total_active_days_to_cutoff,
    ({{CLICKS_PER_DAY}})     AS clicks_per_active_day,
    -- ------------------------------------------------------------------
    -- Trend features. All computed only from weeks inside the window.
    -- A negative click_slope means engagement is decaying.
    -- ------------------------------------------------------------------
    ({{CLICK_DELTA}})        AS click_delta,
    ({{ACTIVE_DELTA}})       AS active_days_delta,
    ({{CLICK_SLOPE}})        AS click_slope,
    -- ------------------------------------------------------------------
    -- Recency
    -- ------------------------------------------------------------------
    ({{CUTOFF_DAY}} - COALESCE(last_active_day_in_window, 0)) AS days_since_last_activity,
    -- ------------------------------------------------------------------
    -- Submission rate recomputed against the calendar that was actually due
    -- ------------------------------------------------------------------
    CASE
        WHEN COALESCE(assessments_due_by_cutoff, 0) > 0
        THEN CAST(COALESCE(submitted_by_cutoff, 0) AS DOUBLE)
             / assessments_due_by_cutoff
    END AS submission_rate,
    -- ------------------------------------------------------------------
    -- Explicit missingness flags.
    -- A NULL mean_score means "no score was observed by the cutoff". That is
    -- NOT the same as a score of zero, and the model is told which is which.
    -- ------------------------------------------------------------------
    (mean_score IS NULL)      AS has_no_score_by_cutoff,
    (COALESCE(submitted_by_cutoff, 0) = 0) AS has_no_submission_by_cutoff,
    (COALESCE(lateness_observations, 0) = 0) AS lateness_unavailable
FROM joined;

-- 03_assessment_features.sql
-- ---------------------------------------------------------------------------
-- Assessment-side inputs for the cutoff-safe feature snapshot.
--
-- Two different things are counted here and they must not be confused:
--
--   1. The assessment CALENDAR for a module-presentation (what was due).
--      This comes from `assessments` and is joined to the cohort, so an
--      assessment the learner never submitted still counts as "due".
--
--   2. The learner's SUBMISSIONS (what they actually did).
--      This comes from `studentAssessment`.
--
-- Lateness is only computed where `assessments.date` is a real day offset.
-- Roughly half of OULAD's Exam assessments record '?' for the due date; those
-- contribute to submission counts but never to a lateness statistic.
-- ---------------------------------------------------------------------------

-- Assessment calendar with a flag for whether a usable due date exists.
CREATE OR REPLACE VIEW assessment_calendar AS
SELECT
    code_module,
    code_presentation,
    id_assessment,
    assessment_type,
    due_day,
    has_due_date,
    weight
FROM stg_assessments;

-- Per-learner-course assessment features, computed for a cutoff week.
-- Placeholders: {{CUTOFF_WEEK}} and {{CUTOFF_DAY}} (last day of that week).
CREATE OR REPLACE VIEW learner_assessment_features AS
WITH learner_keys AS (
    SELECT code_module, code_presentation, id_student FROM stg_student_info
),
-- What was due for this learner by the cutoff (calendar join, not submissions)
due AS (
    SELECT
        l.code_module,
        l.code_presentation,
        l.id_student,
        COUNT(*) AS assessments_due_by_cutoff,
        COUNT(*) FILTER (WHERE c.has_due_date) AS assessments_due_with_date,
        SUM(c.weight) FILTER (WHERE c.has_due_date) AS weight_due_by_cutoff
    FROM learner_keys AS l
    JOIN assessment_calendar AS c
      ON c.code_module = l.code_module
     AND c.code_presentation = l.code_presentation
    WHERE c.due_day IS NOT NULL
      AND c.due_day <= {{CUTOFF_DAY}}
    GROUP BY 1, 2, 3
),
-- What the learner actually submitted by the cutoff
submitted AS (
    SELECT
        code_module,
        code_presentation,
        id_student,
        COUNT(*)                                        AS submitted_by_cutoff,
        COUNT(*) FILTER (WHERE is_banked = 1)           AS banked_submissions,
        AVG(score)                                      AS mean_score,
        MIN(score)                                      AS min_score,
        MAX(score)                                      AS max_score,
        AVG(weight)                                     AS mean_assessment_weight,
        -- Lateness statistics: only rows with a real due date contribute
        COUNT(lateness_days)                            AS lateness_observations,
        AVG(lateness_days)                              AS mean_lateness_days,
        MAX(lateness_days)                              AS max_lateness_days,
        COUNT(*) FILTER (WHERE lateness_days > 0)       AS late_submissions,
        COUNT(*) FILTER (WHERE lateness_days <= 0)      AS on_time_submissions,
        MAX(submitted_day)                              AS last_submission_day,
        MIN(submitted_day)                              AS first_submission_day
    FROM stg_learner_assessment_events
    WHERE submitted_day IS NOT NULL
      AND submitted_day <= {{CUTOFF_DAY}}
    GROUP BY 1, 2, 3
),
-- Score trend: latest submitted score minus the earliest submitted score.
-- Uses a window function so "latest" means latest by submission day.
scored AS (
    SELECT
        code_module,
        code_presentation,
        id_student,
        score,
        submitted_day,
        ROW_NUMBER() OVER (
            PARTITION BY code_module, code_presentation, id_student
            ORDER BY submitted_day ASC, id_assessment ASC
        ) AS rn_first,
        ROW_NUMBER() OVER (
            PARTITION BY code_module, code_presentation, id_student
            ORDER BY submitted_day DESC, id_assessment DESC
        ) AS rn_last
    FROM stg_learner_assessment_events
    WHERE submitted_day IS NOT NULL
      AND submitted_day <= {{CUTOFF_DAY}}
      AND score IS NOT NULL
),
score_edges AS (
    SELECT
        code_module,
        code_presentation,
        id_student,
        MAX(CASE WHEN rn_first = 1 THEN score END) AS first_score,
        MAX(CASE WHEN rn_last  = 1 THEN score END) AS latest_score
    FROM scored
    GROUP BY 1, 2, 3
)
SELECT
    l.code_module,
    l.code_presentation,
    l.id_student,
    COALESCE(d.assessments_due_by_cutoff, 0)     AS assessments_due_by_cutoff,
    COALESCE(d.assessments_due_with_date, 0)     AS assessments_due_with_date,
    d.weight_due_by_cutoff,
    COALESCE(s.submitted_by_cutoff, 0)           AS submitted_by_cutoff,
    COALESCE(s.banked_submissions, 0)            AS banked_submissions,
    s.mean_score,
    s.min_score,
    s.max_score,
    s.mean_assessment_weight,
    s.lateness_observations,
    s.mean_lateness_days,
    s.max_lateness_days,
    s.late_submissions,
    s.on_time_submissions,
    s.last_submission_day,
    s.first_submission_day,
    e.first_score,
    e.latest_score,
    -- Score trend only when at least two scored submissions exist.
    CASE
        WHEN e.first_score IS NOT NULL AND e.latest_score IS NOT NULL
         AND s.submitted_by_cutoff > 1
        THEN e.latest_score - e.first_score
    END AS score_delta,
    -- Submission rate against the assessments actually due by the cutoff.
    CASE
        WHEN COALESCE(d.assessments_due_by_cutoff, 0) > 0
        THEN CAST(COALESCE(s.submitted_by_cutoff, 0) AS DOUBLE)
             / d.assessments_due_by_cutoff
    END AS submission_rate,
    -- Overdue = due by the cutoff but never submitted by the cutoff.
    CASE
        WHEN d.assessments_due_by_cutoff IS NULL THEN NULL
        ELSE GREATEST(
            d.assessments_due_by_cutoff - COALESCE(s.submitted_by_cutoff, 0),
            0
        )
    END AS overdue_count
FROM learner_keys AS l
LEFT JOIN due       AS d USING (code_module, code_presentation, id_student)
LEFT JOIN submitted AS s USING (code_module, code_presentation, id_student)
LEFT JOIN score_edges AS e USING (code_module, code_presentation, id_student);

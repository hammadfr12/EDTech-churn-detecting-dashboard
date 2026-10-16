-- 05_cohort_metrics.sql
-- ---------------------------------------------------------------------------
-- Cohort-level aggregates: the outcome funnel, weekly activity, and
-- week-over-week activity drop-off.
--
-- IMPORTANT: activity drop-off is NOT withdrawal. A fall in clicks is a
-- change in observed behaviour; it is not evidence that a learner has left
-- the course. The dashboard labels these two things separately and this
-- comment exists so nobody quietly conflates them later.
-- ---------------------------------------------------------------------------

-- Enrolled denominator per cohort.
CREATE OR REPLACE VIEW cohort_enrolment AS
SELECT
    code_module,
    code_presentation,
    COUNT(*)                       AS learners_enrolled,
    SUM(CASE WHEN final_result = 'Withdrawn'   THEN 1 ELSE 0 END) AS n_withdrawn,
    SUM(CASE WHEN final_result = 'Fail'        THEN 1 ELSE 0 END) AS n_fail,
    SUM(CASE WHEN final_result = 'Pass'        THEN 1 ELSE 0 END) AS n_pass,
    SUM(CASE WHEN final_result = 'Distinction' THEN 1 ELSE 0 END) AS n_distinction
FROM stg_student_info
GROUP BY 1, 2;

-- Outcome mix at learner-course grain (for the funnel and outcome charts).
CREATE OR REPLACE VIEW cohort_outcomes AS
SELECT
    i.code_module,
    i.code_presentation,
    i.id_student,
    i.final_result,
    c.module_presentation_length,
    -- Course-relative week in which the learner last interacted, if ever.
    MAX(CASE WHEN e.week IS NOT NULL THEN e.week END) AS last_active_week,
    SUM(COALESCE(e.clicks, 0))                        AS total_clicks
FROM stg_student_info AS i
LEFT JOIN stg_courses AS c
       ON c.code_module = i.code_module AND c.code_presentation = i.code_presentation
LEFT JOIN learner_week_engagement AS e
       ON e.code_module = i.code_module
      AND e.code_presentation = i.code_presentation
      AND e.id_student = i.id_student
GROUP BY 1, 2, 3, 4, 5;

-- Weekly activity per cohort. `weeks` is a dense spine built from the maximum
-- observed week in the dataset so cohorts with no activity in a late week
-- still get a row with zero active learners, rather than vanishing.
CREATE OR REPLACE VIEW weekly_cohort_activity AS
WITH week_spine AS (
    SELECT UNNEST(range(1, 30)) AS week
),
cohorts AS (
    SELECT code_module, code_presentation, COUNT(*) AS learners_enrolled
    FROM stg_student_info
    GROUP BY 1, 2
)
SELECT
    co.code_module,
    co.code_presentation,
    sp.week,
    co.learners_enrolled,
    COUNT(DISTINCT e.id_student)                    AS active_learners,
    COALESCE(SUM(e.clicks), 0)                      AS total_clicks,
    CAST(COUNT(DISTINCT e.id_student) AS DOUBLE) / co.learners_enrolled
                                                    AS weekly_active_rate,
    CASE
        WHEN COUNT(DISTINCT e.id_student) > 0
        THEN CAST(COALESCE(SUM(e.clicks), 0) AS DOUBLE)
             / COUNT(DISTINCT e.id_student)
    END                                             AS clicks_per_active_learner
FROM cohorts AS co
CROSS JOIN week_spine AS sp
LEFT JOIN learner_week_engagement AS e
       ON e.code_module = co.code_module
      AND e.code_presentation = co.code_presentation
      AND e.week = sp.week
WHERE sp.week <= 25
GROUP BY 1, 2, 3, 4;

-- Week-over-week activity drop-off, with an explicit denominator and a guard
-- against division by zero.
--
-- Bounded to each cohort's last week that actually had activity. Without that
-- bound the dense week spine produces trailing weeks with zero active learners
-- on both sides of the comparison, which would either be dropped (harmless) or
-- read as a 100% collapse (misleading). A cohort that has genuinely finished
-- is not "dropping off".
CREATE OR REPLACE VIEW activity_dropoff AS
WITH last_active AS (
    SELECT
        code_module,
        code_presentation,
        MAX(week) AS last_active_week
    FROM weekly_cohort_activity
    WHERE active_learners > 0
    GROUP BY 1, 2
),
w AS (
    SELECT
        a.code_module,
        a.code_presentation,
        a.week,
        a.learners_enrolled,
        a.active_learners,
        LAG(a.active_learners) OVER (
            PARTITION BY a.code_module, a.code_presentation ORDER BY a.week
        ) AS prev_active_learners
    FROM weekly_cohort_activity AS a
    JOIN last_active AS l
      ON l.code_module = a.code_module
     AND l.code_presentation = a.code_presentation
    WHERE a.week <= l.last_active_week
)
SELECT
    code_module,
    code_presentation,
    week,
    learners_enrolled,
    prev_active_learners,
    active_learners,
    prev_active_learners - active_learners AS net_change,
    CASE
        WHEN prev_active_learners IS NULL OR prev_active_learners = 0 THEN NULL
        ELSE CAST(prev_active_learners - active_learners AS DOUBLE)
             / prev_active_learners
    END AS dropoff_rate
FROM w
WHERE prev_active_learners IS NOT NULL
  AND prev_active_learners > 0;

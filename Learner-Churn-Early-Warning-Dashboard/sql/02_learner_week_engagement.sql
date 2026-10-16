-- 02_learner_week_engagement.sql
-- ---------------------------------------------------------------------------
-- Builds the learner-course-presentation-week engagement grain from the
-- clickstream, plus a wide week matrix used by the feature snapshot.
--
-- Week mapping (see src/config.py for the canonical definition):
--     week = floor(day / 7) + 1     for day >= 0
-- so week 1 covers days 0-6 inclusive and day 0 belongs to week 1.
--
-- Negative day values are pre-course activity. They are excluded from the
-- weekly series and captured separately in learner_precourse_engagement so
-- the information is available without distorting the week axis.
-- ---------------------------------------------------------------------------

-- Long grain: one row per learner-course-week that has any activity.
CREATE OR REPLACE VIEW learner_week_engagement AS
SELECT
    code_module,
    code_presentation,
    id_student,
    CAST(FLOOR(day / {{DAYS_PER_WEEK}}) + 1 AS INTEGER) AS week,
    SUM(sum_click)      AS clicks,
    COUNT(DISTINCT day) AS active_days,
    MIN(day)            AS first_active_day,
    MAX(day)            AS last_active_day
FROM stg_student_vle
WHERE day IS NOT NULL
  AND day >= 0
GROUP BY 1, 2, 3, 4;

-- Pre-course activity, kept deliberately separate from the week axis.
CREATE OR REPLACE VIEW learner_precourse_engagement AS
SELECT
    code_module,
    code_presentation,
    id_student,
    SUM(sum_click)      AS pre_course_clicks,
    COUNT(DISTINCT day) AS pre_course_active_days,
    MIN(day)            AS earliest_day
FROM stg_student_vle
WHERE day IS NOT NULL
  AND day < 0
GROUP BY 1, 2, 3;

-- Wide matrix: one row per learner-course, weeks 1..MAX_FEATURE_WEEK.
-- A learner-week with no recorded activity is a genuine zero, not a NULL:
-- absence of a clickstream row means zero clicks. This is the opposite of the
-- assessment features, where a missing score is NOT a zero score.
CREATE OR REPLACE VIEW learner_week_matrix AS
SELECT
    code_module,
    code_presentation,
    id_student,
    SUM(CASE WHEN week = 1 THEN clicks      ELSE 0 END) AS clicks_w1,
    SUM(CASE WHEN week = 2 THEN clicks      ELSE 0 END) AS clicks_w2,
    SUM(CASE WHEN week = 3 THEN clicks      ELSE 0 END) AS clicks_w3,
    SUM(CASE WHEN week = 4 THEN clicks      ELSE 0 END) AS clicks_w4,
    SUM(CASE WHEN week = 5 THEN clicks      ELSE 0 END) AS clicks_w5,
    SUM(CASE WHEN week = 6 THEN clicks      ELSE 0 END) AS clicks_w6,
    SUM(CASE WHEN week = 7 THEN clicks      ELSE 0 END) AS clicks_w7,
    SUM(CASE WHEN week = 8 THEN clicks      ELSE 0 END) AS clicks_w8,
    SUM(CASE WHEN week = 1 THEN active_days ELSE 0 END) AS active_days_w1,
    SUM(CASE WHEN week = 2 THEN active_days ELSE 0 END) AS active_days_w2,
    SUM(CASE WHEN week = 3 THEN active_days ELSE 0 END) AS active_days_w3,
    SUM(CASE WHEN week = 4 THEN active_days ELSE 0 END) AS active_days_w4,
    SUM(CASE WHEN week = 5 THEN active_days ELSE 0 END) AS active_days_w5,
    SUM(CASE WHEN week = 6 THEN active_days ELSE 0 END) AS active_days_w6,
    SUM(CASE WHEN week = 7 THEN active_days ELSE 0 END) AS active_days_w7,
    SUM(CASE WHEN week = 8 THEN active_days ELSE 0 END) AS active_days_w8,
    COUNT(DISTINCT CASE WHEN week <= {{MAX_FEATURE_WEEK}} THEN week END) AS active_weeks,
    SUM(clicks) AS total_clicks
FROM learner_week_engagement
WHERE week BETWEEN 1 AND {{MAX_FEATURE_WEEK}}
GROUP BY 1, 2, 3;

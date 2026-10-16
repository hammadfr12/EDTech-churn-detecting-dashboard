-- 01_staging.sql
-- ---------------------------------------------------------------------------
-- Normalises the raw OULAD CSVs into typed, documented staging views.
--
-- Design notes
--   * Every CSV is read with `all_varchar = true` and then cast explicitly
--     with TRY_CAST. OULAD encodes missing values as the literal string '?',
--     so TRY_CAST is the only conversion that is safe for every column:
--     a '?' becomes NULL instead of raising or silently becoming 0.
--   * Text columns that can hold '?' are wrapped in NULLIF(x, '?').
--   * Staging views keep the original grain. No aggregation happens here.
--
-- Placeholder
--   {{RAW}}  directory containing the raw OULAD CSV files.
-- ---------------------------------------------------------------------------

CREATE OR REPLACE VIEW stg_courses AS
SELECT
    code_module,
    code_presentation,
    TRY_CAST(module_presentation_length AS INTEGER) AS module_presentation_length
FROM read_csv_auto('{{RAW}}/courses.csv', header = true, all_varchar = true);

-- One row per learner-course presentation. Grain verified unique by
-- tests/test_join_grain.py.
CREATE OR REPLACE VIEW stg_student_info AS
SELECT
    code_module,
    code_presentation,
    TRY_CAST(id_student AS BIGINT)             AS id_student,
    gender,
    region,
    NULLIF(highest_education, '?')             AS highest_education,
    NULLIF(imd_band, '?')                      AS imd_band,
    NULLIF(age_band, '?')                      AS age_band,
    TRY_CAST(num_of_prev_attempts AS INTEGER)  AS num_of_prev_attempts,
    TRY_CAST(studied_credits AS INTEGER)       AS studied_credits,
    NULLIF(disability, '?')                    AS disability,
    final_result
FROM read_csv_auto('{{RAW}}/studentInfo.csv', header = true, all_varchar = true);

-- Registration dates are day offsets. date_unregistration is '?' for learners
-- who never unregistered. It is NOT used as a feature: unregistration timing
-- is a post-hoc outcome signal (see docs/model_card.md, leakage controls).
CREATE OR REPLACE VIEW stg_student_registration AS
SELECT
    code_module,
    code_presentation,
    TRY_CAST(id_student AS BIGINT)          AS id_student,
    TRY_CAST(date_registration AS INTEGER)  AS date_registration,
    TRY_CAST(date_unregistration AS INTEGER) AS date_unregistration
FROM read_csv_auto('{{RAW}}/studentRegistration.csv', header = true, all_varchar = true);

-- The assessment calendar. `date` is the due date as a day offset; it is '?'
-- for most Exam assessments, which is why lateness can only be computed for
-- the subset of assessments where a real due date exists.
CREATE OR REPLACE VIEW stg_assessments AS
SELECT
    code_module,
    code_presentation,
    TRY_CAST(id_assessment AS BIGINT)   AS id_assessment,
    assessment_type,
    TRY_CAST(date AS INTEGER)           AS due_day,
    TRY_CAST(date AS INTEGER) IS NOT NULL AS has_due_date,
    TRY_CAST(weight AS INTEGER)         AS weight
FROM read_csv_auto('{{RAW}}/assessments.csv', header = true, all_varchar = true);

-- VLE activity definitions (what a site is). week_from / week_to are '?' for
-- most sites and are not used as features.
CREATE OR REPLACE VIEW stg_vle AS
SELECT
    TRY_CAST(id_site AS BIGINT)  AS id_site,
    code_module,
    code_presentation,
    activity_type,
    TRY_CAST(week_from AS INTEGER) AS week_from,
    TRY_CAST(week_to AS INTEGER)   AS week_to
FROM read_csv_auto('{{RAW}}/vle.csv', header = true, all_varchar = true);

-- The raw clickstream. 10.6M rows in the full dataset.
CREATE OR REPLACE VIEW stg_student_vle AS
SELECT
    code_module,
    code_presentation,
    TRY_CAST(id_student AS BIGINT) AS id_student,
    TRY_CAST(id_site AS BIGINT)    AS id_site,
    TRY_CAST(date AS INTEGER)      AS day,
    TRY_CAST(sum_click AS BIGINT)  AS sum_click
FROM read_csv_auto('{{RAW}}/studentVle.csv', header = true, all_varchar = true);

-- Learner submissions joined to the assessment calendar.
-- `lateness_days` is populated ONLY where the assessment genuinely has a due
-- date. It is never imputed or fabricated.
CREATE OR REPLACE VIEW stg_learner_assessment_events AS
SELECT
    a.code_module,
    a.code_presentation,
    TRY_CAST(sa.id_student AS BIGINT)      AS id_student,
    a.id_assessment,
    a.assessment_type,
    a.due_day,
    a.has_due_date,
    a.weight,
    TRY_CAST(sa.date_submitted AS INTEGER) AS submitted_day,
    TRY_CAST(sa.score AS INTEGER)          AS score,
    TRY_CAST(sa.is_banked AS INTEGER)      AS is_banked,
    CASE
        WHEN a.has_due_date AND TRY_CAST(sa.date_submitted AS INTEGER) IS NOT NULL
        THEN TRY_CAST(sa.date_submitted AS INTEGER) - a.due_day
    END AS lateness_days
FROM read_csv_auto('{{RAW}}/studentAssessment.csv', header = true, all_varchar = true) AS sa
JOIN stg_assessments AS a
  ON TRY_CAST(sa.id_assessment AS BIGINT) = a.id_assessment;

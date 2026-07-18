-- int_cohort_health
-- Intermediate: weekly token retention, peak week identification, decay onset.
-- One row per company per week-since-signup.

WITH weekly AS (
    SELECT
        company_id,
        FLOOR(day_since_signup / 7.0)               AS week_num,
        AVG(rolling_7d_avg_tokens)                  AS avg_weekly_tokens,
        MIN(usage_date)                             AS week_start_date
    FROM int_usage_velocity
    GROUP BY 1, 2
),

with_peak AS (
    SELECT
        *,
        -- Peak week: highest avg token volume across all weeks for that company
        MAX(avg_weekly_tokens) OVER (
            PARTITION BY company_id
        )                                           AS peak_weekly_tokens,

        -- Previous week tokens (for retention calc)
        LAG(avg_weekly_tokens, 1) OVER (
            PARTITION BY company_id ORDER BY week_num
        )                                           AS prev_week_tokens
    FROM weekly
),

with_retention AS (
    SELECT
        *,
        -- Retention vs peak (1.0 = at peak, 0.0 = fully churned)
        CASE
            WHEN peak_weekly_tokens IS NULL OR peak_weekly_tokens = 0 THEN NULL
            ELSE ROUND(avg_weekly_tokens / peak_weekly_tokens, 4)
        END                                         AS retention_vs_peak,

        -- WoW retention (vs prior week)
        CASE
            WHEN prev_week_tokens IS NULL OR prev_week_tokens = 0 THEN NULL
            ELSE ROUND(avg_weekly_tokens / prev_week_tokens, 4)
        END                                         AS wow_retention_rate,

        -- Is this the peak week?
        CASE
            WHEN avg_weekly_tokens = MAX(avg_weekly_tokens) OVER (PARTITION BY company_id)
            THEN TRUE ELSE FALSE
        END                                         AS is_peak_week

    FROM with_peak
)

SELECT
    company_id,
    CAST(week_num AS INTEGER)                   AS week_num,
    week_start_date,
    ROUND(avg_weekly_tokens, 0)                 AS avg_weekly_tokens,
    ROUND(peak_weekly_tokens, 0)                AS peak_weekly_tokens,
    retention_vs_peak,
    wow_retention_rate,
    is_peak_week
FROM with_retention
ORDER BY company_id, week_num

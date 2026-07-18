-- fct_cohort_churn
-- Fact table: one row per company. Joins velocity + friction + cohort health.
-- Assigns hard churn label, soft decay label, treatment/control group.

WITH peak_per_company AS (
    SELECT
        company_id,
        MAX(rolling_7d_avg_tokens)              AS peak_rolling_avg
    FROM int_usage_velocity
    GROUP BY company_id
),

peak_with_date AS (
    -- Find the first date where rolling avg hits the peak value
    SELECT
        v.company_id,
        p.peak_rolling_avg,
        MIN(v.usage_date)                       AS peak_date
    FROM int_usage_velocity v
    JOIN peak_per_company p ON v.company_id = p.company_id
    WHERE v.rolling_7d_avg_tokens = p.peak_rolling_avg
    GROUP BY v.company_id, p.peak_rolling_avg
),

churn_onset AS (
    -- First date rolling avg drops >= 80% from peak (hard churn)
    -- and >= 40% from peak (soft decay), both after peak_date
    SELECT
        v.company_id,
        p.peak_rolling_avg,
        MIN(CASE
            WHEN p.peak_rolling_avg > 0
             AND v.rolling_7d_avg_tokens <= p.peak_rolling_avg * 0.20
            THEN v.usage_date END)              AS hard_churn_date,
        MIN(CASE
            WHEN p.peak_rolling_avg > 0
             AND v.rolling_7d_avg_tokens <= p.peak_rolling_avg * 0.60
            THEN v.usage_date END)              AS soft_decay_date
    FROM int_usage_velocity v
    JOIN peak_with_date p ON v.company_id = p.company_id
    WHERE v.usage_date > p.peak_date
    GROUP BY v.company_id, p.peak_rolling_avg
)

SELECT
    f.company_id,
    f.company_name,
    f.industry,
    f.contract_tier,
    f.signup_date,
    f.infra_risk_score,
    f.growth_rate,
    f.friction_bucket,
    f.avg_daily_tokens_onboarding,
    f.avg_ttft_ms_onboarding,
    f.max_ttft_ms_onboarding,
    f.avg_context_saturation_onboarding,
    f.max_context_saturation_onboarding,
    f.avg_cache_ratio_onboarding,
    f.count_429_onboarding,
    f.count_500_onboarding,
    f.count_503_onboarding,
    f.total_errors_onboarding,

    f.is_treatment,
    CASE WHEN f.is_treatment = FALSE THEN TRUE ELSE FALSE END   AS is_control,

    ROUND(p.peak_rolling_avg, 0)                AS peak_rolling_avg_tokens,
    p.peak_date,

    CASE WHEN c.hard_churn_date IS NOT NULL THEN TRUE ELSE FALSE END
                                                AS hard_churn_label,
    CASE WHEN c.soft_decay_date IS NOT NULL THEN TRUE ELSE FALSE END
                                                AS soft_decay_label,

    CASE
        WHEN c.hard_churn_date IS NOT NULL
        THEN CAST(c.hard_churn_date - f.signup_date AS INTEGER)
        ELSE 90
    END                                         AS days_to_hard_churn,

    CASE
        WHEN c.soft_decay_date IS NOT NULL
        THEN CAST(c.soft_decay_date - f.signup_date AS INTEGER)
        ELSE 90
    END                                         AS days_to_soft_decay,

    CASE WHEN c.hard_churn_date IS NOT NULL THEN 1 ELSE 0 END  AS hard_churn_observed,
    CASE WHEN c.soft_decay_date IS NOT NULL THEN 1 ELSE 0 END  AS soft_decay_observed

FROM int_friction_flags         AS f
LEFT JOIN peak_with_date        AS p ON f.company_id = p.company_id
LEFT JOIN churn_onset           AS c ON f.company_id = c.company_id

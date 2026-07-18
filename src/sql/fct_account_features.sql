-- fct_account_features
-- Fact table: one wide row per company with all engineered signals for Stage 5 ML.
-- Extends fct_cohort_churn with post-onboarding velocity trend features.

WITH velocity_trends AS (
    -- Summarise usage velocity across different time windows
    SELECT
        company_id,

        -- Days 0-14: onboarding window avg
        AVG(CASE WHEN day_since_signup BETWEEN 0  AND 13  THEN total_tokens END)
                                                AS avg_tokens_days_0_14,
        -- Days 15-30
        AVG(CASE WHEN day_since_signup BETWEEN 14 AND 29  THEN total_tokens END)
                                                AS avg_tokens_days_15_30,
        -- Days 31-60
        AVG(CASE WHEN day_since_signup BETWEEN 30 AND 59  THEN total_tokens END)
                                                AS avg_tokens_days_31_60,

        -- Onboarding TTFT stats
        AVG(CASE WHEN day_since_signup BETWEEN 0 AND 13 THEN avg_ttft_ms END)
                                                AS avg_ttft_days_0_14,
        MAX(CASE WHEN day_since_signup BETWEEN 0 AND 13 THEN avg_ttft_ms END)
                                                AS max_ttft_days_0_14,

        -- Cache ratio trend
        AVG(CASE WHEN day_since_signup BETWEEN 0  AND 13  THEN prompt_cache_ratio END)
                                                AS avg_cache_ratio_0_14,
        AVG(CASE WHEN day_since_signup BETWEEN 14 AND 29  THEN prompt_cache_ratio END)
                                                AS avg_cache_ratio_15_30,

        -- Context saturation
        AVG(CASE WHEN day_since_signup BETWEEN 0 AND 13 THEN max_context_saturation_pct END)
                                                AS avg_ctx_sat_0_14,
        MAX(CASE WHEN day_since_signup BETWEEN 0 AND 13 THEN max_context_saturation_pct END)
                                                AS max_ctx_sat_0_14,

        -- WoW momentum in onboarding (positive = growing, negative = slowing)
        AVG(CASE WHEN day_since_signup BETWEEN 7 AND 13 THEN wow_pct_change END)
                                                AS avg_wow_pct_change_week2

    FROM int_usage_velocity
    GROUP BY company_id
),

token_growth_slope AS (
    -- Linear trend of daily tokens during onboarding (proxy for adoption momentum)
    -- Positive slope = healthy ramp, negative = early warning
    SELECT
        company_id,
        REGR_SLOPE(total_tokens, day_since_signup) AS token_growth_slope_onboarding
    FROM int_usage_velocity
    WHERE day_since_signup BETWEEN 0 AND 13
    GROUP BY company_id
)

SELECT
    -- Identity & metadata
    ch.company_id,
    ch.company_name,
    ch.industry,
    ch.contract_tier,
    ch.signup_date,
    ch.infra_risk_score,
    ch.growth_rate,
    ch.friction_bucket,
    ch.is_treatment,

    -- Churn labels (ML targets)
    ch.hard_churn_label,
    ch.soft_decay_label,
    ch.days_to_hard_churn,
    ch.days_to_soft_decay,
    ch.hard_churn_observed,
    ch.soft_decay_observed,

    -- Peak stats
    ch.peak_rolling_avg_tokens,

    -- Friction signals (onboarding window)
    ch.count_429_onboarding,
    ch.count_500_onboarding,
    ch.count_503_onboarding,
    ch.total_errors_onboarding,
    ch.avg_ttft_ms_onboarding,
    ch.max_ttft_ms_onboarding,
    ch.avg_context_saturation_onboarding,
    ch.max_context_saturation_onboarding,
    ch.avg_cache_ratio_onboarding,

    -- Velocity features from int_usage_velocity
    ROUND(vt.avg_tokens_days_0_14,  0)          AS avg_tokens_days_0_14,
    ROUND(vt.avg_tokens_days_15_30, 0)          AS avg_tokens_days_15_30,
    ROUND(vt.avg_tokens_days_31_60, 0)          AS avg_tokens_days_31_60,
    ROUND(vt.avg_ttft_days_0_14,    2)          AS avg_ttft_days_0_14,
    ROUND(vt.max_ttft_days_0_14,    2)          AS max_ttft_days_0_14,
    ROUND(vt.avg_cache_ratio_0_14,  4)          AS avg_cache_ratio_0_14,
    ROUND(vt.avg_cache_ratio_15_30, 4)          AS avg_cache_ratio_15_30,
    ROUND(vt.avg_ctx_sat_0_14,      4)          AS avg_ctx_sat_0_14,
    ROUND(vt.max_ctx_sat_0_14,      4)          AS max_ctx_sat_0_14,
    ROUND(vt.avg_wow_pct_change_week2, 2)       AS avg_wow_pct_change_week2,

    -- Growth slope during onboarding
    ROUND(sl.token_growth_slope_onboarding, 2)  AS token_growth_slope_onboarding,

    -- Ratio: tokens week2 / tokens week1 (momentum indicator)
    CASE
        WHEN vt.avg_tokens_days_0_14 IS NULL OR vt.avg_tokens_days_0_14 = 0 THEN NULL
        ELSE ROUND(vt.avg_tokens_days_15_30 / vt.avg_tokens_days_0_14, 4)
    END                                         AS token_momentum_w2_vs_w1

FROM fct_cohort_churn           AS ch
LEFT JOIN velocity_trends       AS vt ON ch.company_id = vt.company_id
LEFT JOIN token_growth_slope    AS sl ON ch.company_id = sl.company_id

-- int_usage_velocity
-- Intermediate: day-over-day token delta, 7-day rolling average, WoW % change.
-- Builds on stg_daily_usage. One row per company per day.

WITH base AS (
    SELECT * FROM stg_daily_usage
),

with_rolling AS (
    SELECT
        *,
        -- 7-day rolling average (current day + 6 prior days)
        AVG(total_tokens) OVER (
            PARTITION BY company_id
            ORDER BY usage_date
            ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
        )                                           AS rolling_7d_avg_tokens,

        -- Previous day tokens (for DoD delta)
        LAG(total_tokens, 1) OVER (
            PARTITION BY company_id ORDER BY usage_date
        )                                           AS prev_day_tokens,

        -- Token volume 7 days ago (for WoW comparison)
        LAG(total_tokens, 7) OVER (
            PARTITION BY company_id ORDER BY usage_date
        )                                           AS tokens_7d_ago

    FROM base
)

SELECT
    log_id,
    usage_date,
    company_id,
    company_name,
    industry,
    contract_tier,
    signup_date,
    day_since_signup,
    is_onboarding_window,
    infra_risk_score,
    growth_rate,

    -- Raw volumes
    input_tokens,
    output_tokens,
    prompt_cached_tokens,
    total_tokens,
    prompt_cache_ratio,
    avg_ttft_ms,
    max_context_saturation_pct,

    -- Velocity metrics
    ROUND(rolling_7d_avg_tokens, 0)             AS rolling_7d_avg_tokens,

    -- Day-over-day absolute delta
    (total_tokens - prev_day_tokens)            AS dod_token_delta,

    -- Day-over-day % change (NULL safe)
    CASE
        WHEN prev_day_tokens IS NULL OR prev_day_tokens = 0 THEN NULL
        ELSE ROUND(
            (CAST(total_tokens AS DOUBLE) - prev_day_tokens) /
            prev_day_tokens * 100, 2)
    END                                         AS dod_pct_change,

    -- Week-over-week % change
    CASE
        WHEN tokens_7d_ago IS NULL OR tokens_7d_ago = 0 THEN NULL
        ELSE ROUND(
            (CAST(total_tokens AS DOUBLE) - tokens_7d_ago) /
            tokens_7d_ago * 100, 2)
    END                                         AS wow_pct_change

FROM with_rolling

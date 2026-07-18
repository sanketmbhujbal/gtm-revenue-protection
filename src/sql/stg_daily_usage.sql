-- stg_daily_usage
-- Staging layer: cast types, rename columns, join company metadata.
-- One row per company per day. This is the base for all downstream models.

SELECT
    du.log_id,
    CAST(du.date AS DATE)                       AS usage_date,
    du.company_id,
    co.company_name,
    co.industry,
    co.contract_tier,
    CAST(co.signup_date AS DATE)                AS signup_date,
    co.growth_rate,
    co.infra_risk_score,

    -- Token volumes
    CAST(du.input_tokens AS INTEGER)            AS input_tokens,
    CAST(du.output_tokens AS INTEGER)           AS output_tokens,
    CAST(du.prompt_cached_tokens AS INTEGER)    AS prompt_cached_tokens,
    (du.input_tokens + du.output_tokens)        AS total_tokens,

    -- Derived: prompt cache efficiency
    ROUND(
        CAST(du.prompt_cached_tokens AS DOUBLE) /
        NULLIF(CAST(du.input_tokens AS DOUBLE), 0),
    4)                                          AS prompt_cache_ratio,

    -- Latency & capacity
    ROUND(du.avg_ttft_ms, 2)                   AS avg_ttft_ms,
    ROUND(du.max_context_saturation_pct, 4)    AS max_context_saturation_pct,

    -- Onboarding flag: first 14 days since signup
    CASE
        WHEN CAST(du.date AS DATE) <= CAST(co.signup_date AS DATE) + INTERVAL '13 days'
        THEN TRUE ELSE FALSE
    END                                         AS is_onboarding_window,

    -- Day number since signup (0-indexed)
    CAST(
        CAST(du.date AS DATE) - CAST(co.signup_date AS DATE)
    AS INTEGER)                                 AS day_since_signup

FROM read_csv_auto('data/raw/daily_usage.csv')  AS du
JOIN read_csv_auto('data/raw/companies.csv')    AS co
  ON du.company_id = co.company_id

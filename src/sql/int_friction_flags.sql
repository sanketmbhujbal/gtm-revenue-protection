WITH onboarding_usage AS (
    SELECT company_id,company_name,industry,contract_tier,infra_risk_score,growth_rate,signup_date,
        COUNT(*) AS onboarding_days_active, AVG(total_tokens) AS avg_daily_tokens_onboarding,
        MAX(total_tokens) AS max_daily_tokens_onboarding, AVG(avg_ttft_ms) AS avg_ttft_ms_onboarding,
        MAX(avg_ttft_ms) AS max_ttft_ms_onboarding, AVG(max_context_saturation_pct) AS avg_context_saturation_onboarding,
        MAX(max_context_saturation_pct) AS max_context_saturation_onboarding, AVG(prompt_cache_ratio) AS avg_cache_ratio_onboarding,
        MIN(usage_date) AS first_usage_date, MAX(usage_date) AS last_onboarding_date
    FROM stg_daily_usage WHERE is_onboarding_window=TRUE GROUP BY 1,2,3,4,5,6,7),
onboarding_errors AS (
    SELECT e.company_id, CAST(e.timestamp AS DATE) AS error_date, e.error_code, COUNT(*) AS error_count
    FROM read_csv_auto('data/raw/api_errors.csv') AS e
    JOIN read_csv_auto('data/raw/companies.csv') AS co ON e.company_id=co.company_id
    WHERE CAST(e.timestamp AS DATE)<=CAST(co.signup_date AS DATE)+INTERVAL '13 days' GROUP BY 1,2,3),
error_pivoted AS (
    SELECT company_id,
        SUM(CASE WHEN error_code=429 THEN error_count ELSE 0 END) AS count_429_onboarding,
        SUM(CASE WHEN error_code=500 THEN error_count ELSE 0 END) AS count_500_onboarding,
        SUM(CASE WHEN error_code=503 THEN error_count ELSE 0 END) AS count_503_onboarding,
        SUM(CASE WHEN error_code=400 THEN error_count ELSE 0 END) AS count_400_onboarding,
        SUM(error_count) AS total_errors_onboarding FROM onboarding_errors GROUP BY 1)
SELECT ou.*,COALESCE(ep.count_429_onboarding,0) AS count_429_onboarding,COALESCE(ep.count_500_onboarding,0) AS count_500_onboarding,
    COALESCE(ep.count_503_onboarding,0) AS count_503_onboarding,COALESCE(ep.count_400_onboarding,0) AS count_400_onboarding,
    COALESCE(ep.total_errors_onboarding,0) AS total_errors_onboarding,
    CASE WHEN infra_risk_score>=0.55 THEN TRUE ELSE FALSE END AS is_treatment,
    CASE WHEN infra_risk_score>=0.55 AND avg_ttft_ms_onboarding>=400 THEN 'Severe'
         WHEN infra_risk_score>=0.55 THEN 'Mild' ELSE 'None' END AS friction_bucket
FROM onboarding_usage AS ou LEFT JOIN error_pivoted AS ep ON ou.company_id=ep.company_id

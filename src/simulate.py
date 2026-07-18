"""
Stage 1: GTM Revenue Protection Engine — Platform Simulator v2
===============================================================
Generates three relational CSVs:
  - data/raw/companies.csv
  - data/raw/daily_usage.csv
  - data/raw/api_errors.csv

Key design changes from v1:
  - Churn is probabilistic, not deterministic. High friction raises
    churn probability but does not guarantee it.
  - Multiple signals contribute independently to churn probability:
    TTFT (40%), rate limits (30%), error rate (20%), context saturation (10%)
  - ~70-75% of friction companies churn; ~10-15% of control companies
    churn from other causes (product fit, budget, etc.)
  - Decay severity varies continuously — not all churners hit the same floor
  - SHAP will show distributed feature importance across all signals

Usage:
  python src/simulate.py                          # defaults: 500 companies, seed=42
  python src/simulate.py --n_companies 300
  python src/simulate.py --seed 99
"""

import argparse
import os
import uuid
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# ── Constants ──────────────────────────────────────────────────────────────

INDUSTRIES     = ["Healthcare", "Finance", "Tech", "Legal"]
CONTRACT_TIERS = ["Enterprise", "Growth"]
ENDPOINTS      = ["/v1/messages", "/v1/embeddings"]
ERROR_CODES    = [429, 500, 503, 400]

TIER_TOKEN_BASE = {
    "Enterprise": (180_000, 60_000),
    "Growth":     (40_000,  15_000),
}

INDUSTRY_MULTIPLIER = {
    "Healthcare": 1.1,
    "Finance":    1.3,
    "Tech":       1.0,
    "Legal":      0.85,
}

ERROR_WEIGHTS = {429: 0.50, 500: 0.25, 503: 0.15, 400: 0.10}

SIMULATION_DAYS   = 90
ONBOARDING_DAYS   = 14
FRICTION_THRESHOLD = 0.55   # infra_risk_score >= this => treatment group

# ── Churn probability weights (must sum to 1.0) ────────────────────────────
# These control how much each signal contributes to churn probability.
# TTFT is the strongest signal but not the only one.
SIGNAL_WEIGHTS = {
    "ttft":              0.40,
    "rate_limits":       0.30,
    "error_rate":        0.20,
    "context_saturation":0.10,
}

# Base churn probability for control group (churn from non-friction causes)
CONTROL_BASE_CHURN_PROB = 0.06  # base non-friction churn (budget, product fit)
# Sigmoid params: midpoint=0.44, scale=13, max_contrib=0.70
# See compute_churn_probability() for details


# ── Helper functions ────────────────────────────────────────────────────────

def new_uuid() -> str:
    return str(uuid.uuid4())


def ramp_factor(day: int, growth_rate: float = 0.6) -> float:
    """Sigmoid ramp: ~0.3 on day 1, ~1.0 by day 20-30."""
    return 0.3 + 0.7 / (1 + np.exp(-growth_rate * (day - 10)))


def weekend_dip(date: datetime) -> float:
    return 0.65 if date.weekday() >= 5 else 1.0


def compute_churn_probability(
    ttft_score: float,         # 0.0 (fast) to 1.0 (very slow)
    rate_limit_score: float,   # 0.0 (none) to 1.0 (many)
    error_rate_score: float,   # 0.0 (clean) to 1.0 (error-prone)
    ctx_sat_score: float,      # 0.0 (low) to 1.0 (saturated)
    base_prob: float,          # base churn probability for this company
) -> float:
    """
    Sigmoid mapping of weighted friction score -> churn probability.

    Uses a sigmoid rather than a linear mapping so that:
    - Low friction scores (control group, avg ~0.26) map to low churn (~8-12%)
    - High friction scores (treatment group, avg ~0.52) map to high churn (~55-70%)
    - The transition is sharp but not binary — some high-friction companies survive

    Expected outcomes at n=500, seed=42:
    - Treatment group churn: ~55-65%
    - Control group churn:   ~12-18%
    - XGBoost ROC-AUC:       ~0.82-0.88
    """
    friction_score = (
        SIGNAL_WEIGHTS["ttft"]               * ttft_score +
        SIGNAL_WEIGHTS["rate_limits"]        * rate_limit_score +
        SIGNAL_WEIGHTS["error_rate"]         * error_rate_score +
        SIGNAL_WEIGHTS["context_saturation"] * ctx_sat_score
    )
    # Sigmoid squashing: midpoint=0.44, scale=13
    # At friction_score=0.26 (control avg): contrib ~0.08
    # At friction_score=0.52 (treatment avg): contrib ~0.60
    sigmoid_contrib = 1.0 / (1.0 + np.exp(-13.0 * (friction_score - 0.44)))
    churn_prob = base_prob + 0.70 * sigmoid_contrib
    return float(np.clip(churn_prob, 0.0, 1.0))


def apply_decay(base_volume: float, day: int, decay_onset: int,
                decay_rate: float) -> float:
    """
    Exponential decay after decay_onset.
    decay_rate varies per company — some churn fast, some slowly.
    Floor at 3% of base (company hasn't fully disappeared).
    """
    if day < decay_onset:
        return base_volume
    days_into_decay = day - decay_onset
    decay_factor = np.exp(-decay_rate * days_into_decay)
    return base_volume * max(decay_factor, 0.03)


# ── Stage 1A: Generate companies ───────────────────────────────────────────

def generate_companies(n: int, rng: np.random.Generator,
                        generation_start: datetime) -> pd.DataFrame:
    """
    Each company gets a set of latent friction signal scores drawn from
    probability distributions. These drive all downstream behaviour.
    """
    rows = []
    for i in range(n):
        tier     = rng.choice(CONTRACT_TIERS, p=[0.35, 0.65])
        industry = rng.choice(INDUSTRIES)

        # infra_risk_score: hidden variable, beta distributed
        if tier == "Enterprise":
            infra_risk = float(np.clip(rng.beta(1.5, 4.0), 0, 1))
        else:
            infra_risk = float(np.clip(rng.beta(2.0, 3.0), 0, 1))

        # ── Latent signal scores — each driven by infra_risk with added noise
        # Weights tuned so treatment avg friction ~0.52, control avg ~0.26
        # This produces treatment churn ~55-65%, control ~12-18% via sigmoid

        # TTFT: strongest infra_risk signal (0.85 weight)
        ttft_score = float(np.clip(
            infra_risk * 0.85 + rng.beta(1.2, 4.0) * 0.15, 0, 1))

        # Rate limits: moderate infra_risk correlation (0.70 weight)
        rate_limit_score = float(np.clip(
            infra_risk * 0.70 + rng.beta(1.5, 5.0) * 0.30, 0, 1))

        # Error rate: weaker correlation — more environmental noise (0.55 weight)
        error_rate_score = float(np.clip(
            infra_risk * 0.55 + rng.beta(1.2, 6.0) * 0.45, 0, 1))

        # Context saturation: mostly usage-pattern driven, weak infra correlation
        ctx_sat_score = float(np.clip(
            infra_risk * 0.20 + rng.beta(1.5, 5.0) * 0.80, 0, 1))

        # Base churn probability (non-friction causes: budget, product fit, etc.)
        base_churn_prob = 0.06 + rng.uniform(-0.02, 0.02)

        # Final churn probability
        churn_prob = compute_churn_probability(
            ttft_score, rate_limit_score, error_rate_score,
            ctx_sat_score, base_churn_prob)

        # Realise churn: stochastic draw
        will_churn = bool(rng.random() < churn_prob)

        # Decay onset: friction churners decay sooner on average
        # Control churners decay later (other causes take longer to manifest)
        if will_churn and infra_risk >= FRICTION_THRESHOLD:
            decay_onset = int(rng.integers(18, 40))   # friction: early decay
        elif will_churn:
            decay_onset = int(rng.integers(45, 80))   # control churner: late decay
        else:
            decay_onset = SIMULATION_DAYS + 1          # survivor: no decay

        # Decay rate: varies so not all churners hit the same floor
        decay_rate = float(rng.uniform(0.03, 0.09)) if will_churn else 0.0

        growth_rate = float(rng.uniform(0.3, 1.2))
        days_offset = int(rng.integers(0, 30))
        signup_date = generation_start + timedelta(days=days_offset)

        rows.append({
            "company_id":         new_uuid(),
            "company_name":       f"Company_{i+1:04d}",
            "industry":           industry,
            "contract_tier":      tier,
            "signup_date":        signup_date.strftime("%Y-%m-%d"),
            "growth_rate":        round(growth_rate, 4),
            "infra_risk_score":   round(infra_risk, 4),
            # Latent scores (used for signal generation below)
            "_ttft_score":        round(ttft_score, 4),
            "_rate_limit_score":  round(rate_limit_score, 4),
            "_error_rate_score":  round(error_rate_score, 4),
            "_ctx_sat_score":     round(ctx_sat_score, 4),
            "_churn_prob":        round(churn_prob, 4),
            "_will_churn":        will_churn,
            "_decay_onset":       decay_onset,
            "_decay_rate":        round(decay_rate, 4),
        })

    return pd.DataFrame(rows)


# ── Stage 1B: Generate daily usage ─────────────────────────────────────────

def generate_daily_usage(companies: pd.DataFrame, rng: np.random.Generator,
                          generation_start: datetime) -> pd.DataFrame:
    rows = []

    for _, co in companies.iterrows():
        base_mean, base_std = TIER_TOKEN_BASE[co["contract_tier"]]
        base_mean *= INDUSTRY_MULTIPLIER[co["industry"]]

        is_treatment = co["infra_risk_score"] >= FRICTION_THRESHOLD
        will_churn   = co["_will_churn"]
        decay_onset  = co["_decay_onset"]
        decay_rate   = co["_decay_rate"]

        # TTFT parameters driven by ttft_score (not just binary treatment)
        # High ttft_score => draw from high-latency distribution
        ttft_mean_log = 5.1 + co["_ttft_score"] * 2.0   # ranges from ~180ms to ~1500ms
        ttft_sigma    = 0.35 + co["_ttft_score"] * 0.15

        # Context saturation baseline driven by ctx_sat_score
        ctx_sat_base = 0.2 + co["_ctx_sat_score"] * 0.6

        for day in range(SIMULATION_DAYS):
            date  = generation_start + timedelta(days=day)
            ramp  = ramp_factor(day, growth_rate=co["growth_rate"])
            w_dip = weekend_dip(date)

            volume = rng.normal(base_mean * ramp * w_dip, base_std * 0.2)
            volume = max(volume, 1_000)

            # Apply decay if churning
            if will_churn:
                volume = apply_decay(volume, day, decay_onset, decay_rate)

            input_tokens         = int(volume)
            output_tokens        = int(volume * rng.uniform(0.3, 0.7))
            prompt_cached_tokens = int(input_tokens * rng.uniform(0.0, 0.4))

            # TTFT: elevated during onboarding for high-ttft_score companies
            # (not just binary treatment — continuous signal)
            if day < ONBOARDING_DAYS:
                avg_ttft = float(np.clip(
                    rng.lognormal(mean=ttft_mean_log, sigma=ttft_sigma), 50, 3000))
            else:
                # Post-onboarding: some improvement assumed (infra teams fix issues)
                post_mean = ttft_mean_log * 0.75
                avg_ttft  = float(np.clip(
                    rng.lognormal(mean=post_mean, sigma=ttft_sigma), 50, 2000))

            # Context saturation: correlated with ctx_sat_score + volume
            ctx_sat = float(np.clip(
                ctx_sat_base * rng.uniform(0.7, 1.2) +
                (input_tokens / (base_mean * 2.0)) * 0.2,
                0.0, 1.0))

            rows.append({
                "log_id":                     new_uuid(),
                "date":                       date.strftime("%Y-%m-%d"),
                "company_id":                 co["company_id"],
                "input_tokens":               input_tokens,
                "output_tokens":              output_tokens,
                "prompt_cached_tokens":       prompt_cached_tokens,
                "avg_ttft_ms":                round(avg_ttft, 2),
                "max_context_saturation_pct": round(ctx_sat, 4),
            })

    return pd.DataFrame(rows)


# ── Stage 1C: Generate API errors ──────────────────────────────────────────

def generate_api_errors(companies: pd.DataFrame, daily_usage: pd.DataFrame,
                         rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    error_codes  = list(ERROR_WEIGHTS.keys())
    error_probs  = list(ERROR_WEIGHTS.values())

    # Pre-index daily_usage for speed
    usage_by_company = {
        cid: grp for cid, grp in daily_usage.groupby("company_id")}

    for _, co in companies.iterrows():
        if co["company_id"] not in usage_by_company:
            continue

        co_usage = usage_by_company[co["company_id"]]
        all_dates = co_usage["date"].tolist()

        # Onboarding window dates
        min_date = pd.to_datetime(co_usage["date"].min())
        onboarding_cutoff = (min_date + timedelta(days=ONBOARDING_DAYS - 1)).strftime("%Y-%m-%d")
        onboarding_dates  = set(co_usage[co_usage["date"] <= onboarding_cutoff]["date"].tolist())

        # Error rate driven by _error_rate_score (continuous, not binary)
        base_rate = 0.3 + co["_error_rate_score"] * 4.0

        # 429 rate driven by _rate_limit_score
        rate_limit_rate = 0.1 + co["_rate_limit_score"] * 3.5

        for date_str in all_dates:
            date          = datetime.strptime(date_str, "%Y-%m-%d")
            in_onboarding = date_str in onboarding_dates

            # General errors
            n_errors = int(rng.poisson(base_rate))
            for _ in range(n_errors):
                code = int(rng.choice(error_codes, p=error_probs))
                # Exclude 429s here — handled separately below
                if code == 429:
                    code = int(rng.choice([500, 503, 400], p=[0.55, 0.35, 0.10]))
                ts = date.replace(
                    hour=int(rng.integers(0, 24)),
                    minute=int(rng.integers(0, 60)),
                    second=int(rng.integers(0, 60)))
                rows.append({
                    "error_id":   new_uuid(),
                    "timestamp":  ts.strftime("%Y-%m-%d %H:%M:%S"),
                    "company_id": co["company_id"],
                    "error_code": code,
                    "endpoint":   rng.choice(ENDPOINTS),
                })

            # 429s: rate driven by _rate_limit_score, concentrated in onboarding
            rl_multiplier = 2.0 if in_onboarding else 0.5
            n_429 = int(rng.poisson(rate_limit_rate * rl_multiplier))
            for _ in range(n_429):
                ts = date.replace(
                    hour=int(rng.integers(0, 24)),
                    minute=int(rng.integers(0, 60)),
                    second=int(rng.integers(0, 60)))
                rows.append({
                    "error_id":   new_uuid(),
                    "timestamp":  ts.strftime("%Y-%m-%d %H:%M:%S"),
                    "company_id": co["company_id"],
                    "error_code": 429,
                    "endpoint":   rng.choice(ENDPOINTS),
                })

    return pd.DataFrame(rows)


# ── Strip internal columns before writing ──────────────────────────────────

def clean_companies(companies: pd.DataFrame) -> pd.DataFrame:
    """Remove internal simulation columns (prefixed with _) before saving."""
    internal = [c for c in companies.columns if c.startswith("_")]
    return companies.drop(columns=internal)


# ── Validation ─────────────────────────────────────────────────────────────

def validate_outputs(companies: pd.DataFrame, daily_usage: pd.DataFrame,
                      api_errors: pd.DataFrame) -> None:
    print("\n── Validation ────────────────────────────────────")

    dupe_check = daily_usage.duplicated(subset=["company_id", "date"]).sum()
    print(f"  daily_usage company x date uniqueness: {'PASS' if dupe_check == 0 else f'FAIL ({dupe_check})'}")

    orphans = set(daily_usage["company_id"]) - set(companies["company_id"])
    print(f"  daily_usage orphan company_ids: {'PASS' if not orphans else f'FAIL ({len(orphans)})'}")

    print(f"\n── Row counts ────────────────────────────────────")
    print(f"  companies:    {len(companies):>8,}")
    print(f"  daily_usage:  {len(daily_usage):>8,}  (expect {len(companies) * SIMULATION_DAYS:,})")
    print(f"  api_errors:   {len(api_errors):>8,}")

    friction_cos = companies[companies["infra_risk_score"] >= FRICTION_THRESHOLD]
    n_friction   = len(friction_cos)
    n_control    = len(companies) - n_friction
    print(f"\n── Group split ───────────────────────────────────")
    print(f"  Friction (treatment): {n_friction:>5} companies ({n_friction/len(companies)*100:.1f}%)")
    print(f"  Control:              {n_control:>5} companies ({n_control/len(companies)*100:.1f}%)")


def print_churn_summary(companies_with_internal: pd.DataFrame) -> None:
    """Print churn rate by group using internal _will_churn flag."""
    print(f"\n── Churn summary (ground truth) ──────────────────")
    for is_treatment, label in [(True, "Treatment (friction)"), (False, "Control")]:
        mask  = companies_with_internal["infra_risk_score"] >= FRICTION_THRESHOLD
        group = companies_with_internal[mask if is_treatment else ~mask]
        n_churn = group["_will_churn"].sum()
        print(f"  {label:<22}: {n_churn:>3}/{len(group)} churned "
              f"({n_churn/len(group)*100:.1f}%)")

    # Signal score ranges
    print(f"\n── Signal score distributions ────────────────────")
    for col, label in [("_ttft_score","TTFT"), ("_rate_limit_score","Rate limits"),
                        ("_error_rate_score","Error rate"), ("_ctx_sat_score","Context sat")]:
        t = companies_with_internal[companies_with_internal["infra_risk_score"] >= FRICTION_THRESHOLD][col]
        c = companies_with_internal[companies_with_internal["infra_risk_score"] <  FRICTION_THRESHOLD][col]
        print(f"  {label:<14} Treatment mean={t.mean():.2f}  Control mean={c.mean():.2f}")


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="GTM Revenue Protection — Stage 1: Platform Simulator v2")
    parser.add_argument("--n_companies", type=int, default=500)
    parser.add_argument("--seed",        type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    generation_start = datetime(2024, 1, 1)

    print("GTM Revenue Protection Engine — Stage 1: Platform Simulator v2")
    print(f"  Companies:  {args.n_companies}")
    print(f"  Seed:       {args.seed}")
    print(f"  Start date: {generation_start.strftime('%Y-%m-%d')}")
    print(f"  Days:       {SIMULATION_DAYS}")
    print()

    print("Generating companies...")
    companies_full = generate_companies(args.n_companies, rng, generation_start)

    print("Generating daily usage...")
    daily_usage = generate_daily_usage(companies_full, rng, generation_start)

    print("Generating API errors...")
    api_errors = generate_api_errors(companies_full, daily_usage, rng)

    # Write outputs — strip internal columns from companies.csv
    os.makedirs("data/raw", exist_ok=True)
    companies_clean = clean_companies(companies_full)
    companies_clean.to_csv("data/raw/companies.csv",    index=False)
    daily_usage.to_csv("data/raw/daily_usage.csv",      index=False)
    api_errors.to_csv("data/raw/api_errors.csv",         index=False)
    print("\nFiles written to data/raw/")

    validate_outputs(companies_clean, daily_usage, api_errors)
    print_churn_summary(companies_full)
    print("\nStage 1 complete.\n")


if __name__ == "__main__":
    main()

"""
Stage 4: Causal Validation — Script version
============================================
Runs PSM + Kaplan-Meier, saves matched cohort and all figures.
Called by run_pipeline.py.

Can also be run standalone:
    python src/causal_validation.py
"""

import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

os.makedirs('outputs/figures', exist_ok=True)
os.makedirs('data/processed', exist_ok=True)

ACCENT = '#1F4E79'
BLUE2  = '#2E75B6'
RED    = '#C0392B'
GREEN  = '#2E8B57'

plt.rcParams.update({
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'axes.spines.top': False, 'axes.spines.right': False,
    'font.family': 'sans-serif',
})

CALIPER = 0.05


def compute_smd(df_t, df_c, col):
    m_t, m_c = df_t[col].mean(), df_c[col].mean()
    s_t, s_c = df_t[col].std(),  df_c[col].std()
    pooled   = np.sqrt((s_t**2 + s_c**2) / 2)
    return abs((m_t - m_c) / pooled) if pooled > 0 else 0.0


def main():
    print("Stage 4: Causal Validation (PSM + Kaplan-Meier)")

    features_df = pd.read_csv('data/processed/fct_account_features.csv')
    churn_df    = pd.read_csv('data/processed/fct_cohort_churn.csv')

    # ── PSM feature prep ─────────────────────────────────────────────────────
    industry_dummies = pd.get_dummies(features_df['industry'], prefix='ind', drop_first=False)
    features_df = pd.concat([features_df.reset_index(drop=True),
                              industry_dummies.reset_index(drop=True)], axis=1)
    features_df['is_enterprise']            = (features_df['contract_tier'] == 'Enterprise').astype(int)
    features_df['log_avg_tokens_onboarding'] = np.log1p(features_df['avg_tokens_days_0_14'].fillna(0))

    ind_cols       = [c for c in features_df.columns if c.startswith('ind_')]
    covariate_cols = ['is_enterprise', 'log_avg_tokens_onboarding', 'growth_rate'] + ind_cols
    X_raw = features_df[covariate_cols].fillna(0)
    y     = features_df['is_treatment'].astype(int)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_scaled, y)
    features_df['propensity_score'] = lr.predict_proba(X_scaled)[:, 1]

    # ── Matching ──────────────────────────────────────────────────────────────
    caliper      = CALIPER * features_df['propensity_score'].std()
    treatment_df = features_df[features_df['is_treatment']].copy().reset_index(drop=True)
    control_df   = features_df[~features_df['is_treatment']].copy().reset_index(drop=True)
    matched_pairs, used = [], set()

    for _, t_row in treatment_df.sort_values('propensity_score').iterrows():
        avail = control_df[~control_df.index.isin(used)].copy()
        avail['ps_diff'] = abs(avail['propensity_score'] - t_row['propensity_score'])
        cands = avail[avail['ps_diff'] <= caliper]
        if len(cands) == 0:
            continue
        best = cands.loc[cands['ps_diff'].idxmin()]
        used.add(best.name)
        matched_pairs.append({
            'treatment_company_id': t_row['company_id'],
            'control_company_id':   best['company_id'],
            'ps_diff': abs(t_row['propensity_score'] - best['propensity_score']),
        })

    matched_df = pd.DataFrame(matched_pairs)
    n_matched  = len(matched_df)
    print(f"  Matched pairs: {n_matched} / {len(treatment_df)}")

    # ── SMD table ─────────────────────────────────────────────────────────────
    matched_t_ids = matched_df['treatment_company_id'].tolist()
    matched_c_ids = matched_df['control_company_id'].tolist()
    post_t = features_df[features_df['company_id'].isin(matched_t_ids)]
    post_c = features_df[features_df['company_id'].isin(matched_c_ids)]

    smd_rows = []
    for col in covariate_cols:
        smd_pre  = compute_smd(treatment_df, control_df, col)
        smd_post = compute_smd(post_t, post_c, col)
        smd_rows.append({'covariate': col, 'smd_before': round(smd_pre,3),
                         'smd_after': round(smd_post,3)})
    smd_table = pd.DataFrame(smd_rows)
    max_smd   = smd_table['smd_after'].max()
    print(f"  Max post-match SMD: {max_smd:.3f}  (target < 0.10)")
    print(f"  All balanced (< 0.10): {(smd_table['smd_after'] < 0.10).all()}")
    if max_smd > 0.10:
        print("  Note: residual imbalance expected — friction signals explain treatment")
        print("  assignment but including them collapses common support.")

    # ── SMD love plot ─────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    y_pos = np.arange(len(smd_table))
    ax.barh(y_pos - 0.2, smd_table['smd_before'], 0.35, color=RED,   alpha=0.7, label='Before')
    ax.barh(y_pos + 0.2, smd_table['smd_after'],  0.35, color=GREEN, alpha=0.7, label='After')
    ax.axvline(0.10, color='gray', lw=1.5, linestyle='--', label='SMD=0.10')
    ax.set_yticks(y_pos); ax.set_yticklabels(smd_table['covariate'], fontsize=9)
    ax.set_xlabel('Standardized Mean Difference'); ax.invert_yaxis()
    ax.set_title('Covariate Balance: Before vs After PSM')
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig('outputs/figures/psm_smd_loveplot.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: outputs/figures/psm_smd_loveplot.png")

    # ── Save matched cohort ───────────────────────────────────────────────────
    churn_t = churn_df[churn_df['company_id'].isin(matched_t_ids)][
        ['company_id','days_to_hard_churn','hard_churn_observed',
         'days_to_soft_decay','soft_decay_observed','is_treatment']].copy()
    churn_c = churn_df[churn_df['company_id'].isin(matched_c_ids)][
        ['company_id','days_to_hard_churn','hard_churn_observed',
         'days_to_soft_decay','soft_decay_observed','is_treatment']].copy()
    matched_cohort = pd.concat([churn_t, churn_c], ignore_index=True)
    matched_cohort.to_csv('data/processed/psm_matched_cohort.csv', index=False)

    # ── Kaplan-Meier ──────────────────────────────────────────────────────────
    t_group = matched_cohort[matched_cohort['is_treatment'] == True]
    c_group = matched_cohort[matched_cohort['is_treatment'] == False]

    kmf_t = KaplanMeierFitter()
    kmf_c = KaplanMeierFitter()
    kmf_t.fit(t_group['days_to_hard_churn'], t_group['hard_churn_observed'], label='Treatment (friction)')
    kmf_c.fit(c_group['days_to_hard_churn'], c_group['hard_churn_observed'], label='Control (no friction)')

    lr_result = logrank_test(
        t_group['days_to_hard_churn'], c_group['days_to_hard_churn'],
        event_observed_A=t_group['hard_churn_observed'],
        event_observed_B=c_group['hard_churn_observed'])

    p_val = lr_result.p_value
    p_label = 'p < 0.001' if p_val < 0.001 else f'p = {p_val:.4f}'
    print(f"  Log-rank {p_label}  |  Significant: {p_val < 0.05}")

    fig, ax = plt.subplots(figsize=(11, 5.5))
    kmf_t.plot_survival_function(ax=ax, color=RED,   ci_show=True, ci_alpha=0.12)
    kmf_c.plot_survival_function(ax=ax, color=BLUE2, ci_show=True, ci_alpha=0.12)
    ax.axhline(0.5, color='gray', lw=1.2, linestyle='--', alpha=0.6)
    ax.text(0.97, 0.97, p_label, transform=ax.transAxes, ha='right', va='top',
            fontsize=11, fontweight='bold', color=ACCENT,
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#EEF4FB', edgecolor=ACCENT, lw=1.2))
    med_t = kmf_t.median_survival_time_
    if not np.isinf(med_t):
        ax.axvline(med_t, color=RED, lw=1.2, linestyle=':', alpha=0.7)
        ax.text(med_t+1, 0.08, f'Median\n{med_t:.0f}d', color=RED, fontsize=8.5)
    ax.set_title('Token Retention Survival Curve — PSM Matched Cohort\n(Hard Churn: >=80% drop from peak)')
    ax.set_xlabel('Days Since Signup'); ax.set_ylabel('Probability of Retention')
    ax.set_ylim(-0.05, 1.05); ax.set_xlim(0, 92)
    ax.legend(frameon=False, loc='lower left')
    plt.tight_layout()
    plt.savefig('outputs/figures/km_curves_hard_churn.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: outputs/figures/km_curves_hard_churn.png")

    print(f"\nStage 4 complete.")
    print(f"  Matched pairs: {n_matched}  |  Log-rank {p_label}")
    print(f"  Median survival (treatment): day {med_t:.0f}" if not np.isinf(med_t) else "  Median survival (treatment): >90 days")


if __name__ == '__main__':
    main()

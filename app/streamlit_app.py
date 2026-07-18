"""
Stage 6: GTM Revenue Protection Engine — Streamlit Dashboard v2
================================================================
Tab 1 — AE Alert Feed: high-risk accounts, cohort heatmap, SHAP chart
Tab 2 — Account Drilldown: usage trend, errors, TTFT, SHAP, recommendation
Tab 3 — Counterfactual Simulator: what-if infrastructure improvement scenarios

Run from project root:
    streamlit run app/streamlit_app.py
"""

import os, sys
import pickle
import warnings
warnings.filterwarnings('ignore')
import subprocess
from pathlib import Path
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import LabelEncoder

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GTM Revenue Protection Engine",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Always resolve project root relative to this file
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
os.chdir(PROJECT_ROOT)
 
# Auto-run pipeline if processed data doesn't exist
if not (PROJECT_ROOT / "data" / "processed" / "scored_accounts.csv").exists():
    with st.spinner("First run: generating data and running pipeline (~60 seconds)..."):
        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "run_pipeline.py")],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
        )
    if result.returncode != 0:
        st.error("Pipeline failed. See details below.")
        st.code(result.stderr or result.stdout)
        st.stop()
    st.rerun()

ACCENT = "#1F4E79"
BLUE2  = "#2E75B6"
RED    = "#C0392B"
AMBER  = "#E67E22"
GREEN  = "#27AE60"

st.markdown("""
<style>
    .main { background-color: #F8F9FA; }
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
    h1 { color: #1F4E79; }
    h2, h3 { color: #2E75B6; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background-color: none;
        border-radius: 6px 6px 0 0;
        padding: 8px 20px;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)


# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    scored  = pd.read_csv("data/processed/scored_accounts.csv")
    cf      = pd.read_csv("data/processed/counterfactual_scenarios.csv")
    churn   = pd.read_csv("data/processed/fct_cohort_churn.csv")
    usage   = pd.read_csv("data/processed/stg_daily_usage.csv",
                           parse_dates=["usage_date", "signup_date"])
    errors  = pd.read_csv("data/raw/api_errors.csv", parse_dates=["timestamp"])
    return scored, cf, churn, usage, errors

@st.cache_resource
def load_model():
    with open("models/xgb_churn_model.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_features():
    df = pd.read_csv("data/processed/fct_account_features.csv")
    le_ind  = LabelEncoder().fit(df["industry"])
    le_tier = LabelEncoder().fit(df["contract_tier"])
    df["industry_enc"]      = le_ind.transform(df["industry"])
    df["contract_tier_enc"] = le_tier.transform(df["contract_tier"])
    feature_cols = [
        "count_429_onboarding","count_500_onboarding","count_503_onboarding",
        "total_errors_onboarding","avg_ttft_ms_onboarding","max_ttft_ms_onboarding",
        "avg_ttft_days_0_14","max_ttft_days_0_14","avg_context_saturation_onboarding",
        "max_context_saturation_onboarding","avg_cache_ratio_onboarding",
        "avg_cache_ratio_0_14","avg_ctx_sat_0_14","max_ctx_sat_0_14",
        "avg_tokens_days_0_14","avg_wow_pct_change_week2",
        "token_growth_slope_onboarding","growth_rate",
        "industry_enc","contract_tier_enc",
    ]
    X = df[feature_cols].fillna(0)
    return X, df, feature_cols

scored, cf_df, churn_df, usage_df, errors_df = load_data()
model = load_model()
X_all, feat_df, feature_cols = load_features()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# 🛡️ GTM Revenue Protection Engine")
st.markdown("**LLM API Platform · Enterprise Account Health & Risk Intelligence**")
st.divider()

tab1, tab2, tab3 = st.tabs([
    "AE Alert Feed",
    "Account Drilldown",
    "Counterfactual Simulator",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — AE ALERT FEED
# ══════════════════════════════════════════════════════════════════════════════
with tab1:

    high_risk   = scored[scored["churn_probability"] >= 0.75]
    critical    = scored[scored["risk_tier"] == "Critical"]
    arr_at_risk = int(high_risk["contract_tier"].map(
        {"Enterprise": 120_000, "Growth": 24_000}).sum())

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Accounts Monitored", f"{len(scored):,}")
    k2.metric("High-Risk Accounts (≥ 0.75)", f"{len(high_risk):,}",
              delta=f"{len(high_risk)/len(scored)*100:.1f}% of portfolio",
              delta_color="inverse")
    k3.metric("Critical Risk (≥ 0.85)", f"{len(critical):,}")
    k4.metric("Est. ARR at Risk", f"${arr_at_risk:,.0f}",
              delta="from high-risk accounts", delta_color="inverse")

    st.divider()

    # Filters
    col_f1, col_f2, col_f3 = st.columns([2, 2, 2])
    with col_f1:
        risk_threshold = st.slider("Risk score threshold", 0.50, 0.99, 0.75, 0.05)
    with col_f2:
        tier_filter = st.multiselect("Contract tier",
            options=["Enterprise", "Growth"], default=["Enterprise", "Growth"])
    with col_f3:
        industry_filter = st.multiselect("Industry",
            options=sorted(scored["industry"].unique()),
            default=sorted(scored["industry"].unique()))

    filtered = scored[
        (scored["churn_probability"] >= risk_threshold) &
        (scored["contract_tier"].isin(tier_filter)) &
        (scored["industry"].isin(industry_filter))
    ].copy()

    st.markdown(f"### 🚨 High-Risk Account Feed  —  {len(filtered)} accounts")
    st.caption("Select an account name in the **Account Drilldown** tab for a full usage and risk breakdown.")

    if len(filtered) == 0:
        st.info("No accounts match the current filters.")
    else:
        def risk_badge(score):
            if score >= 0.85: return f"🔴 {score:.2f}"
            elif score >= 0.75: return f"🟠 {score:.2f}"
            else: return f"🟡 {score:.2f}"

        display = filtered[[
            "company_name","contract_tier","industry",
            "churn_probability","top_friction_trigger",
            "recommended_action","action_owner",
            "count_429_onboarding","avg_ttft_ms_onboarding",
        ]].copy()
        display["churn_probability"] = display["churn_probability"].apply(risk_badge)
        display["avg_ttft_ms_onboarding"] = display["avg_ttft_ms_onboarding"].round(0).astype(int).astype(str) + " ms"
        display["count_429_onboarding"]   = display["count_429_onboarding"].astype(int)
        display.columns = ["Company","Tier","Industry","Risk Score",
                           "Primary Friction Trigger","Recommended Action","Owner",
                           "429s (onboarding)","Avg TTFT"]
        st.dataframe(display, width='stretch', height=300, hide_index=True)

    st.divider()

    # Cohort heatmap + SHAP side by side
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown("### 🗺️ Cohort Heatmap · Token Velocity by Friction Bucket")

        # Compute key stat for annotation
        usage_with_bucket = usage_df.merge(
            churn_df[["company_id","friction_bucket"]], on="company_id")
        bins   = [0,14,28,42,56,70,90]
        labels = ["Days 0-14","Days 15-28","Days 29-42","Days 43-56","Days 57-70","Days 71-90"]
        usage_with_bucket["day_bin"] = pd.cut(
            usage_with_bucket["day_since_signup"], bins=bins, labels=labels, right=False)
        heatmap_data = (
            usage_with_bucket
            .groupby(["friction_bucket","day_bin"], observed=True)["total_tokens"]
            .mean().unstack("day_bin")
            .reindex(["Severe","Mild","None"]).reindex(columns=labels)
        )

        # Executive annotation
        severe_early = heatmap_data.loc["Severe","Days 0-14"] if "Severe" in heatmap_data.index else None
        severe_late  = heatmap_data.loc["Severe","Days 57-70"] if "Severe" in heatmap_data.index else None
        none_early   = heatmap_data.loc["None","Days 0-14"]
        none_late    = heatmap_data.loc["None","Days 57-70"]

        if severe_early and severe_late:
            pct_drop = (1 - severe_late / severe_early) * 100
            ctrl_growth = (none_late / none_early - 1) * 100
            st.info(
                f"📌 **Key finding:** Companies with severe onboarding friction show "
                f"**{pct_drop:.0f}% lower token velocity by days 57–70**, "
                f"while healthy accounts grow **{ctrl_growth:.0f}%** over the same window."
            )

        heatmap_norm = heatmap_data.div(heatmap_data.iloc[:, 0], axis=0)
        fig_heat = px.imshow(
            heatmap_norm,
            labels=dict(x="Usage Window", y="Friction Bucket", color="Relative Volume"),
            color_continuous_scale="RdYlGn",
            zmin=0, zmax=2,
            aspect="auto",
            text_auto=".2f",
        )
        fig_heat.update_layout(
            margin=dict(l=10,r=10,t=10,b=10), height=220,
            font=dict(size=11), coloraxis_showscale=False)
        fig_heat.update_traces(textfont_size=10)
        st.plotly_chart(fig_heat, width='stretch')
        st.caption("Values relative to each group's Days 0–14 baseline · Green = growing · Red = declining")

    with col_right:
        st.markdown("### 📊 Top Risk Factors (SHAP)")
        st.caption("Mean absolute SHAP value across all accounts")
        shap_path = "outputs/figures/shap_barchart.png"
        if os.path.exists(shap_path):
            st.image(shap_path, width='stretch')
        else:
            st.info("Run `python src/score.py` to generate SHAP chart.")

    st.divider()

    # Portfolio risk distribution
    st.markdown("### 📈 Portfolio Risk Distribution")
    col_d1, col_d2 = st.columns(2)

    with col_d1:
        fig_hist = px.histogram(
            scored, x="churn_probability", nbins=30,
            color_discrete_sequence=[ACCENT],
            labels={"churn_probability": "Churn Probability", "count": "Accounts"},
            title="Distribution of Churn Risk Scores",
        )
        fig_hist.add_vline(x=0.75, line_dash="dash", line_color=AMBER,
                           annotation_text="Alert threshold (0.75)")
        fig_hist.update_layout(
            margin=dict(l=10,r=10,t=40,b=10), height=300,
            showlegend=False, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_hist, width='stretch')

    with col_d2:
        industry_risk = (
            scored.groupby("industry")["churn_probability"]
            .mean().reset_index().sort_values("churn_probability", ascending=True))
        fig_ind = px.bar(
            industry_risk, x="churn_probability", y="industry",
            orientation="h", color_discrete_sequence=[BLUE2],
            labels={"churn_probability": "Avg Churn Probability", "industry": ""},
            title="Avg Risk Score by Industry",
        )
        fig_ind.update_layout(
            margin=dict(l=10,r=10,t=40,b=10), height=300,
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_ind, width='stretch')


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — ACCOUNT DRILLDOWN
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### Account Drilldown")

    # Account selector — show high-risk first
    all_accounts = scored.sort_values("churn_probability", ascending=False)
    account_options = all_accounts.apply(
        lambda r: f"{r['company_name']}  ({r['churn_probability']:.2f} risk · {r['contract_tier']})",
        axis=1).tolist()
    account_ids = all_accounts["company_id"].tolist()

    selected_label = st.selectbox(
        "Select account", options=account_options,
        help="Accounts sorted by risk score; highest risk first")

    selected_idx  = account_options.index(selected_label)
    selected_id   = account_ids[selected_idx]
    selected_row  = all_accounts.iloc[selected_idx]

    st.divider()

    # ── Identity & risk header ────────────────────────────────────────────────
    h1, h2, h3, h4, h5 = st.columns(5)
    h1.metric("Company",        selected_row["company_name"])
    h2.metric("Contract Tier",  selected_row["contract_tier"])
    h3.metric("Industry",       selected_row["industry"])
    h4.metric("Churn Risk",     f"{selected_row['churn_probability']:.2f}",
              delta="Critical" if selected_row["churn_probability"] >= 0.85 else "High" \
                    if selected_row["churn_probability"] >= 0.75 else "Medium",
              delta_color="inverse" if selected_row["churn_probability"] >= 0.75 else "normal")
    h5.metric("Friction Bucket", selected_row["friction_bucket"])

    st.divider()

    # ── Prescription box ──────────────────────────────────────────────────────
    trigger  = selected_row["top_friction_trigger"]
    action   = selected_row["recommended_action"]
    owner    = selected_row["action_owner"]

    col_p1, col_p2 = st.columns([1, 2])
    with col_p1:
        st.error(f"**⚠️ Primary Signal**\n\n{trigger}")
    with col_p2:
        st.success(f"**✅ Recommended Action** *(Owner: {owner})*\n\n{action}")

    st.divider()

    # ── Usage trend + Error timeline ──────────────────────────────────────────
    co_usage  = usage_df[usage_df["company_id"] == selected_id].sort_values("usage_date")
    co_errors = errors_df[errors_df["company_id"] == selected_id].copy()

    if len(co_usage) > 0:
        col_u1, col_u2 = st.columns(2)

        with col_u1:
            st.markdown("#### Token Velocity (90 days)")
            fig_usage = go.Figure()
            fig_usage.add_trace(go.Scatter(
                x=co_usage["usage_date"],
                y=co_usage["total_tokens"],
                mode="lines",
                line=dict(color=ACCENT, width=1.5),
                name="Daily tokens",
                opacity=0.5,
            ))
            # 7-day rolling average
            co_usage["rolling_7d"] = co_usage["total_tokens"].rolling(7, min_periods=1).mean()
            fig_usage.add_trace(go.Scatter(
                x=co_usage["usage_date"],
                y=co_usage["rolling_7d"],
                mode="lines",
                line=dict(color=RED, width=2.5),
                name="7-day avg",
            ))
            # Shade onboarding window
            onboard_end = co_usage["signup_date"].iloc[0] + pd.Timedelta(days=14)
            fig_usage.add_vrect(
                x0=co_usage["usage_date"].min(), x1=onboard_end,
                fillcolor="rgba(230,126,34,0.10)", line_width=0,
                annotation_text="Onboarding", annotation_position="top left",
                annotation_font_size=10,
            )
            fig_usage.update_layout(
                margin=dict(l=10,r=10,t=10,b=10), height=280,
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                legend=dict(orientation="h", y=1.1),
                xaxis_title="", yaxis_title="Total Tokens",
                yaxis=dict(tickformat=".2s"),
            )
            st.plotly_chart(fig_usage, width='stretch')

        with col_u2:
            st.markdown("#### TTFT During Onboarding (ms)")
            onboard_usage = co_usage[co_usage["day_since_signup"] <= 13]
            if len(onboard_usage) > 0:
                # Portfolio avg TTFT for comparison
                portfolio_avg_ttft = usage_df[usage_df["day_since_signup"] <= 13]["avg_ttft_ms"].mean()
                fig_ttft = go.Figure()
                fig_ttft.add_hline(
                    y=portfolio_avg_ttft, line_dash="dash",
                    line_color="gray", opacity=0.6,
                    annotation_text=f"Portfolio avg ({portfolio_avg_ttft:.0f}ms)",
                    annotation_position="bottom right",
                    annotation_font_size=10,
                )
                fig_ttft.add_trace(go.Bar(
                    x=onboard_usage["day_since_signup"],
                    y=onboard_usage["avg_ttft_ms"],
                    marker_color=[
                        RED if v > portfolio_avg_ttft * 1.5 else BLUE2
                        for v in onboard_usage["avg_ttft_ms"]
                    ],
                    name="Avg TTFT",
                ))
                fig_ttft.update_layout(
                    margin=dict(l=10,r=10,t=10,b=10), height=280,
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    xaxis_title="Day Since Signup", yaxis_title="Avg TTFT (ms)",
                    showlegend=False,
                )
                st.plotly_chart(fig_ttft, width='stretch')
            else:
                st.info("No onboarding TTFT data available.")

    # ── Error breakdown ───────────────────────────────────────────────────────
    st.markdown("#### API Error Events")
    if len(co_errors) > 0:
        col_e1, col_e2 = st.columns(2)

        with col_e1:
            # Error count by type
            error_counts = co_errors["error_code"].value_counts().reset_index()
            error_counts.columns = ["Error Code","Count"]
            error_counts["Error Code"] = error_counts["Error Code"].astype(str)
            color_map = {"429": RED, "500": AMBER, "503": AMBER, "400": BLUE2}
            fig_err = px.bar(
                error_counts, x="Error Code", y="Count",
                color="Error Code",
                color_discrete_map=color_map,
                title="Errors by Type (full 90 days)",
            )
            fig_err.update_layout(
                margin=dict(l=10,r=10,t=40,b=10), height=250,
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                showlegend=False,
            )
            st.plotly_chart(fig_err, width='stretch')

        with col_e2:
            # 429s over time
            errors_429 = co_errors[co_errors["error_code"] == 429].copy()
            if len(errors_429) > 0:
                errors_429["date"] = errors_429["timestamp"].dt.date
                daily_429 = errors_429.groupby("date").size().reset_index(name="count")
                daily_429["date"] = pd.to_datetime(daily_429["date"])
                fig_429 = px.bar(
                    daily_429, x="date", y="count",
                    color_discrete_sequence=[RED],
                    title="Daily HTTP 429 (Rate Limit) Events",
                )
                fig_429.update_layout(
                    margin=dict(l=10,r=10,t=40,b=10), height=250,
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    xaxis_title="", yaxis_title="429 Events",
                )
                st.plotly_chart(fig_429, width='stretch')
            else:
                st.info("No 429 errors for this account.")
    else:
        st.info("No API error events recorded for this account.")

    # ── Key onboarding metrics ────────────────────────────────────────────────
    st.markdown("#### Onboarding Signal Summary")
    feat_row = feat_df[feat_df["company_id"] == selected_id]
    if len(feat_row) > 0:
        fr = feat_row.iloc[0]
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("429s (onboarding)",    int(fr["count_429_onboarding"]))
        m2.metric("Avg TTFT (onboarding)",f"{fr['avg_ttft_ms_onboarding']:.0f} ms")
        m3.metric("Max Context Sat.",     f"{fr['max_context_saturation_onboarding']:.0%}")
        m4.metric("Cache Ratio",          f"{fr['avg_cache_ratio_onboarding']:.0%}")
        m5.metric("Token Growth Slope",   f"{fr['token_growth_slope_onboarding']:+.0f}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — COUNTERFACTUAL SIMULATOR
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### What-If Infrastructure Improvement Simulator")
    st.markdown(
        "Perturb friction features and re-score all accounts with the trained XGBoost model "
        "to estimate ARR retained from accounts moving below the 0.75 risk threshold."
    )
    st.caption("ARR assumptions: Enterprise = `$120K` · Growth = `$24K` · Model: XGBoost, trained on 14-day onboarding signals")
    st.divider()

    col_s1, col_s2, col_s3 = st.columns([2, 2, 2])
    with col_s1:
        ttft_reduction = st.slider("TTFT reduction (%)", 0, 90, 50, 5)
    with col_s2:
        rl_reduction = st.slider("Rate limit reduction (%)", 0, 100, 0, 10)
    with col_s3:
        arr_enterprise = st.number_input("Avg ARR — Enterprise ($)", value=120_000, step=10_000)
        arr_growth     = st.number_input("Avg ARR — Growth ($)",     value=24_000,  step=5_000)

    # Live re-scoring
    baseline_proba = model.predict_proba(X_all)[:, 1]

    X_cf = X_all.copy()
    for col in [c for c in X_cf.columns if "ttft" in c.lower()]:
        X_cf[col] = X_cf[col] * (1 - ttft_reduction / 100)
    for col in [c for c in X_cf.columns if "429" in c]:
        X_cf[col] = X_cf[col] * (1 - rl_reduction / 100)

    cf_proba   = model.predict_proba(X_cf)[:, 1]
    arr_map    = {"Enterprise": arr_enterprise, "Growth": arr_growth}
    was_high   = baseline_proba >= 0.75
    now_safe   = cf_proba < 0.75
    saved_mask = was_high & now_safe
    n_saved    = int(saved_mask.sum())
    arr_retained = int(sum(
        arr_map.get(feat_df.iloc[i]["contract_tier"], arr_growth)
        for i in np.where(saved_mask)[0]
    ))

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("TTFT Reduction",        f"{ttft_reduction}%")
    r2.metric("Rate Limit Reduction",  f"{rl_reduction}%")
    r3.metric("Accounts Moved to Safe",f"{n_saved:,}",
              delta=f"from {int(was_high.sum())} high-risk", delta_color="normal")
    r4.metric("Est. ARR Retained",     f"${arr_retained:,.0f}", delta_color="normal")

    st.divider()

    col_cf1, col_cf2 = st.columns(2)

    with col_cf1:
        st.markdown("**Risk Score Distribution: Before vs After**")
        fig_compare = go.Figure()
        fig_compare.add_trace(go.Histogram(
            x=baseline_proba, name="Baseline",
            opacity=0.65, marker_color=RED, nbinsx=30))
        fig_compare.add_trace(go.Histogram(
            x=cf_proba, name="Counterfactual",
            opacity=0.65, marker_color=GREEN, nbinsx=30))
        fig_compare.add_vline(x=0.75, line_dash="dash", line_color=AMBER,
                               annotation_text="Alert threshold")
        fig_compare.update_layout(
            barmode="overlay",
            xaxis_title="Churn Probability", yaxis_title="Accounts",
            legend=dict(orientation="h", y=1.12),
            margin=dict(l=10,r=10,t=20,b=10), height=320,
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_compare, width='stretch')

    with col_cf2:
        st.markdown("**Pre-Built Scenario Comparison**")
        st.caption("Model-derived estimates from counterfactual_scenarios.csv")
        cf_display = cf_df[["scenario","accounts_moved_to_safe","estimated_arr_retained"]].copy()
        cf_display["estimated_arr_retained"] = cf_display["estimated_arr_retained"].apply(lambda x: f"${x:,.0f}")
        cf_display.columns = ["Scenario","Accounts Saved","Est. ARR Retained"]
        st.dataframe(cf_display, width='stretch', hide_index=True, height=240)

    if n_saved > 0:
        st.divider()
        st.markdown(f"#### Accounts Moving to Safe  (n={n_saved})")
        saved_idx = np.where(saved_mask)[0]
        saved_df  = feat_df.iloc[saved_idx][["company_name","contract_tier","industry"]].copy()
        saved_df["Baseline Risk"] = baseline_proba[saved_idx].round(3)
        saved_df["CF Risk"]       = cf_proba[saved_idx].round(3)
        saved_df["Risk Delta"]    = (baseline_proba[saved_idx] - cf_proba[saved_idx]).round(3)
        saved_df["ARR Value"]     = saved_df["contract_tier"].map(arr_map).apply(lambda x: f"${x:,.0f}")
        saved_df = saved_df.sort_values("Baseline Risk", ascending=False).reset_index(drop=True)
        saved_df.columns = ["Company","Tier","Industry","Baseline Risk","CF Risk","Risk Delta","ARR Value"]
        st.dataframe(saved_df, width='stretch', hide_index=True, height=300)
    else:
        st.info("No accounts move below the 0.75 threshold under current settings. Try increasing TTFT reduction.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "GTM Revenue Protection Engine · Portfolio Project · "
    "Stack: Python · DuckDB · XGBoost · SHAP · Streamlit · "
    "Data: Synthetic (500 companies · 90 days · seed=42)"
)

"""
GTM Revenue Protection Engine — Full Pipeline Runner
=====================================================
Runs all six stages end-to-end from a clean state.

Usage:
    python run_pipeline.py
    python run_pipeline.py --n_companies 200 --seed 99
    python run_pipeline.py --skip_stage1
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path


def run(cmd: list, label: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    start  = time.time()
    result = subprocess.run(cmd, check=False)
    elapsed = time.time() - start
    if result.returncode != 0:
        print(f"\n  FAILED (exit code {result.returncode}). Stopping.")
        sys.exit(result.returncode)
    print(f"\n  Completed in {elapsed:.1f}s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_companies", type=int, default=500)
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--skip_stage1", action="store_true")
    args = parser.parse_args()

    print("\nGTM Revenue Protection Engine — Pipeline")
    print(f"  Companies: {args.n_companies}  |  Seed: {args.seed}\n")
    total_start = time.time()

    if not args.skip_stage1:
        run([sys.executable, "src/simulate.py",
             "--n_companies", str(args.n_companies), "--seed", str(args.seed)],
            "Stage 1: Platform Simulator")
    else:
        print("\nStage 1: Skipped")

    run([sys.executable, "src/run_sql_models.py"],
        "Stage 2: Analytics Engineering (SQL)")

    run([sys.executable, "src/behavior_analysis.py"],
        "Stage 3: Behavior Analysis")

    run([sys.executable, "src/causal_validation.py"],
        "Stage 4: Causal Validation (PSM + Kaplan-Meier)")

    run([sys.executable, "src/score.py"],
        "Stage 5: Prediction + Prescription (XGBoost + SHAP)")

    elapsed = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"  Pipeline complete in {elapsed:.1f}s")
    print(f"{'='*60}")
    print(f"\n  Key outputs:")
    print(f"    data/processed/scored_accounts.csv")
    print(f"    data/processed/counterfactual_scenarios.csv")
    print(f"    outputs/figures/  — all charts")
    print(f"    models/xgb_churn_model.pkl")
    print(f"\n  Launch dashboard:")
    print(f"    streamlit run app/streamlit_app.py\n")


if __name__ == "__main__":
    main()

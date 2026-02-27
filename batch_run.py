"""
batch_run.py: Batch Runner for the Dynamic-BABE Model
Executes BATCH_ITERATIONS independent simulation runs, sweeping
across both `enable_bridge = False` (Control) and `enable_bridge = True`
(Treatment) conditions, then exports all model-level time-series
data to CSV for statistical analysis.

Output files:
    model_data_bridge_off.csv
    model_data_bridge_on.csv
    agent_data_bridge_off.csv
    agent_data_bridge_on.csv

Run:
    python batch_run.py
"""

import os
import time

import numpy as np
import pandas as pd

import config as cfg
from model import SocialNetworkModel


def run_single(
    run_id,
    enable_bridge,
    seed,
    max_steps=cfg.MAX_STEPS,
):
    """
    Execute one full simulation and return (model_df, agent_df).
    Each row of model_df is one tick; agent_df has one row per
    agent per tick.
    """
    model = SocialNetworkModel(
        num_agents=cfg.NUM_AGENTS,
        ba_m=cfg.BA_EDGE_PARAM,
        enable_bridge=enable_bridge,
        seed=seed,
    )

    for _ in range(max_steps):
        if not model.running:
            break
        model.step()

    # Collect DataFrames
    model_df = model.datacollector.get_model_vars_dataframe().reset_index()
    model_df.rename(columns={"index": "Step"}, inplace=True)
    model_df["Run"] = run_id
    model_df["Bridge"] = enable_bridge

    agent_df = model.datacollector.get_agent_vars_dataframe().reset_index()
    agent_df["Run"] = run_id
    agent_df["Bridge"] = enable_bridge

    return model_df, agent_df


def batch_run(
    iterations=cfg.BATCH_ITERATIONS,
    enable_bridge=False,
    base_seed=cfg.RANDOM_SEED,
):
    """
    Run *iterations* independent simulations and concatenate results.
    Each run receives a unique deterministic seed derived from
    base_seed + run_id so the entire batch is reproducible.
    """
    model_frames = []
    agent_frames = []

    tag = "Bridge ON" if enable_bridge else "Bridge OFF"

    for i in range(iterations):
        seed = base_seed + i
        m_df, a_df = run_single(i, enable_bridge, seed)
        model_frames.append(m_df)
        agent_frames.append(a_df)

        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{tag}] Run {i + 1:>4}/{iterations} complete")

    return pd.concat(model_frames, ignore_index=True), pd.concat(
        agent_frames, ignore_index=True
    )


def main():
    out_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(out_dir, exist_ok=True)

    wall_start = time.perf_counter()

    # ── Condition A: Bridge OFF (Control) ─────────────────────────
    print("━" * 55)
    print("  CONDITION A — Bridge Algorithm DISABLED (Control)")
    print("━" * 55)
    model_off, agent_off = batch_run(
        iterations=cfg.BATCH_ITERATIONS,
        enable_bridge=False,
    )

    # ── Condition B: Bridge ON (Treatment) ────────────────────────
    print()
    print("━" * 55)
    print("  CONDITION B — Bridge Algorithm ENABLED (Treatment)")
    print("━" * 55)
    model_on, agent_on = batch_run(
        iterations=cfg.BATCH_ITERATIONS,
        enable_bridge=True,
    )

    # ── Export ─────────────────────────────────────────────────────
    model_off.to_csv(os.path.join(out_dir, "model_data_bridge_off.csv"), index=False)
    model_on.to_csv(os.path.join(out_dir, "model_data_bridge_on.csv"), index=False)
    agent_off.to_csv(os.path.join(out_dir, "agent_data_bridge_off.csv"), index=False)
    agent_on.to_csv(os.path.join(out_dir, "agent_data_bridge_on.csv"), index=False)

    elapsed = time.perf_counter() - wall_start

    print()
    print("╔══════════════════════════════════════════════════╗")
    print(f"║  Batch complete — {cfg.BATCH_ITERATIONS * 2} total runs in {elapsed:.1f}s")
    print(f"║  Output →  {out_dir}")
    print("╚══════════════════════════════════════════════════╝")

    # ── Quick summary statistics ──────────────────────────────────
    for label, df in [("Bridge OFF", model_off), ("Bridge ON", model_on)]:
        final = df.groupby("Run").last()
        print(f"\n  {label} — Final-step averages across {len(final)} runs:")
        print(f"    Polarization   = {final['Polarization'].mean():.4f}  "
              f"(± {final['Polarization'].std():.4f})")
        print(f"    Revenue        = {final['Revenue'].mean():.2f}  "
              f"(± {final['Revenue'].std():.2f})")
        print(f"    Churn Rate     = {final['Churn_Rate'].mean():.4f}  "
              f"(± {final['Churn_Rate'].std():.4f})")
        print(f"    Avg Frustration= {final['Avg_Frustration'].mean():.4f}  "
              f"(± {final['Avg_Frustration'].std():.4f})")
        print(f"    Active Users   = {final['Active_Users'].mean():.1f}  "
              f"(± {final['Active_Users'].std():.1f})")


if __name__ == "__main__":
    main()

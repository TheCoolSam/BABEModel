"""
batch_run.py: Parallel Batch Runner for the Dynamic-BABE Model
Executes BATCH_ITERATIONS independent simulation runs across a 2×2 factorial
design of: Bridge Algorithm (OFF/ON) × Dyadic Trust (OFF/ON).

Conditions:
    1. Baseline    - Bridge OFF, Trust OFF
    2. Bridge Only - Bridge ON,  Trust OFF
    3. Trust Only  - Bridge OFF, Trust ON
    4. Full Model  - Bridge ON,  Trust ON

Exports all model-level and agent-level time-series data to CSVs.

Uses multiprocessing to run simulations across all available CPU cores.
Each run is fully independent (own seed), so parallelism is trivial.

Output files:
    model_data_baseline.csv, agent_data_baseline.csv
    model_data_bridge_only.csv, agent_data_bridge_only.csv
    model_data_trust_only.csv, agent_data_trust_only.csv
    model_data_full_model.csv, agent_data_full_model.csv

Run:
    python batch_run.py
"""

import os
import time
import multiprocessing as mp
from functools import partial

import numpy as np
import pandas as pd

import config as cfg
from model import SocialNetworkModel


def run_single(
    run_id,
    enable_bridge,
    enable_trust,
    seed,
    max_steps=cfg.MAX_STEPS,
):
    """
    Execute one full simulation and return (model_df, agent_df).
    Each row of model_df is one tick; agent_df has one row per
    agent per tick.

    This function is designed to be called in a worker process —
    it imports nothing that can't be pickled and returns plain
    DataFrames.
    """
    model = SocialNetworkModel(
        num_agents=cfg.NUM_AGENTS,
        ba_m=cfg.BA_EDGE_PARAM,
        enable_bridge=enable_bridge,
        enable_trust=enable_trust,
        seed=seed,
    )

    for _ in range(max_steps):
        if not model.running:
            break
        model.step()

    # Determine condition name
    if enable_bridge and enable_trust:
        cond_name = "Full Model"
    elif enable_bridge:
        cond_name = "Bridge Only"
    elif enable_trust:
        cond_name = "Trust Only"
    else:
        cond_name = "Baseline"

    # Collect DataFrames
    model_df = model.datacollector.get_model_vars_dataframe().reset_index()
    model_df.rename(columns={"index": "Step"}, inplace=True)
    model_df["Run"] = run_id
    model_df["Bridge"] = enable_bridge
    model_df["Trust"] = enable_trust
    model_df["Condition"] = cond_name

    agent_df = model.datacollector.get_agent_vars_dataframe().reset_index()
    agent_df["Run"] = run_id
    agent_df["Bridge"] = enable_bridge
    agent_df["Trust"] = enable_trust
    agent_df["Condition"] = cond_name

    return model_df, agent_df


def _worker(args):
    """
    Thin wrapper for multiprocessing — unpacks the argument tuple
    and calls run_single. Returns (run_id, model_df, agent_df).

    Wrapped in try/except so one failed run doesn't kill the batch.
    """
    run_id, enable_bridge, enable_trust, seed = args
    try:
        m_df, a_df = run_single(run_id, enable_bridge, enable_trust, seed)
        return (run_id, m_df, a_df)
    except Exception as e:
        print(f"  [ERROR] Run {run_id} failed: {e}")
        return (run_id, None, None)


def batch_run(
    iterations=cfg.BATCH_ITERATIONS,
    enable_bridge=False,
    enable_trust=False,
    base_seed=cfg.RANDOM_SEED,
    n_workers=None,
):
    """
    Run *iterations* independent simulations in parallel and
    concatenate results.

    Each run receives a unique deterministic seed derived from
    base_seed + run_id so the entire batch is reproducible
    regardless of execution order or number of workers.
    """
    if n_workers is None:
        n_workers = mp.cpu_count()

    if enable_bridge and enable_trust:
        tag = "Full Model"
    elif enable_bridge:
        tag = "Bridge Only"
    elif enable_trust:
        tag = "Trust Only"
    else:
        tag = "Baseline"

    print(f"  [{tag}] Launching {iterations} runs across {n_workers} cores...")

    # Build argument list: (run_id, enable_bridge, enable_trust, seed)
    args_list = [
        (i, enable_bridge, enable_trust, base_seed + i)
        for i in range(iterations)
    ]

    model_frames = []
    agent_frames = []
    completed = 0

    # Use multiprocessing Pool for parallel execution
    with mp.Pool(processes=n_workers) as pool:
        for run_id, m_df, a_df in pool.imap_unordered(_worker, args_list):
            if m_df is not None:
                model_frames.append(m_df)
                agent_frames.append(a_df)
            completed += 1
            if completed % 10 == 0 or completed == 1 or completed == iterations:
                print(f"  [{tag}] {completed:>4}/{iterations} runs complete")

    return pd.concat(model_frames, ignore_index=True), pd.concat(
        agent_frames, ignore_index=True
    )


def main():
    import argparse

    parser = argparse.ArgumentParser(description="2×2 factorial batch runner")
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Parallel workers (default: all CPU cores). Use 2 for laptop-safe runs.",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=None,
        help=f"Replications per condition (default: {cfg.BATCH_ITERATIONS})",
    )
    args = parser.parse_args()

    out_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(out_dir, exist_ok=True)

    n_cores = args.workers if args.workers is not None else mp.cpu_count()
    iterations = args.iters if args.iters is not None else cfg.BATCH_ITERATIONS
    wall_start = time.perf_counter()

    print(f"  Using {n_cores} workers ({mp.cpu_count()} CPUs available)")
    print(f"  Iterations per condition: {iterations}")
    print()

    conditions = [
        ("Baseline", False, False, "model_data_baseline.csv", "agent_data_baseline.csv"),
        ("Bridge Only", True, False, "model_data_bridge_only.csv", "agent_data_bridge_only.csv"),
        ("Trust Only", False, True, "model_data_trust_only.csv", "agent_data_trust_only.csv"),
        ("Full Model", True, True, "model_data_full_model.csv", "agent_data_full_model.csv"),
    ]

    results = {}

    for name, bridge, trust, m_file, a_file in conditions:
        print("=" * 55)
        print(f"  CONDITION: {name}")
        print("=" * 55)
        model_df, agent_df = batch_run(
            iterations=iterations,
            enable_bridge=bridge,
            enable_trust=trust,
            n_workers=n_cores,
        )
        
        # Save to CSV
        model_df.to_csv(os.path.join(out_dir, m_file), index=False)
        agent_df.to_csv(os.path.join(out_dir, a_file), index=False)
        
        results[name] = model_df
        print()

    elapsed = time.perf_counter() - wall_start
    total_runs = iterations * len(conditions)

    print()
    print("=" * 55)
    print(f"  Batch complete -- {total_runs} total runs in {elapsed:.1f}s")
    print(f"  ({n_cores} workers, ~{elapsed / total_runs:.2f}s per run)")
    print(f"  Output ->  {out_dir}")
    print("=" * 55)

    # Quick summary statistics
    for name, bridge, trust, m_file, a_file in conditions:
        df = results[name]
        final = df.groupby("Run").last()
        print(f"\n  {name} -- Final-step averages across {len(final)} runs:")
        print(f"    Polarization   = {final['Polarization'].mean():.4f}  "
              f"(+/- {final['Polarization'].std():.4f})")
        if "Polarization_All" in final.columns:
            print(f"    Polarization_All= {final['Polarization_All'].mean():.4f}  "
                  f"(+/- {final['Polarization_All'].std():.4f})")
        print(f"    Revenue        = {final['Revenue'].mean():.2f}  "
              f"(+/- {final['Revenue'].std():.2f})")
        print(f"    Churn Rate     = {final['Churn_Rate'].mean():.4f}  "
              f"(+/- {final['Churn_Rate'].std():.4f})")
        print(f"    Avg Frustration= {final['Avg_Frustration'].mean():.4f}  "
              f"(+/- {final['Avg_Frustration'].std():.4f})")
        print(f"    Active Users   = {final['Active_Users'].mean():.1f}  "
              f"(+/- {final['Active_Users'].std():.1f})")


if __name__ == "__main__":
    main()

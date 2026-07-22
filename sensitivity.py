"""
sensitivity.py — One-Factor-At-a-Time (OFAT) Sensitivity Analysis
==================================================================
Systematically sweeps each key parameter across a plausible range
while holding all others at their default values.  For each sweep
point, runs BATCH_ITERATIONS simulations per condition (Bridge OFF/ON)
and records final-step KPIs.

This is standard practice for theoretical ABMs where empirical
calibration data is not available — we demonstrate that the model's
qualitative findings are robust across a wide parameter space.

Output:
    ./output/sensitivity_results.csv
    ./figures/sensitivity_<param>.png  (one per parameter)

Run:
    python sensitivity.py
    python sensitivity.py --quick     (reduced iterations for testing)
"""

import os
import sys
import time
import multiprocessing as mp
from collections import OrderedDict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import config as cfg
from model import SocialNetworkModel

sns.set_theme(style="whitegrid", font_scale=1.1)

OUT_DIR = os.path.join(os.path.dirname(__file__), "output")
FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")


# ====================================================================
# Parameter sweep definitions
# ====================================================================

# Each entry: (config attribute name, display label, list of values)
SWEEP_PARAMS = OrderedDict([
    ("BETA_MEAN",              ("Cognitive Entrenchment (β)",
                                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])),
    ("ASSIMILATION_THRESHOLD", ("Assimilation Threshold",
                                [0.1, 0.2, 0.3, 0.4, 0.5])),
    ("CHURN_THRESHOLD",        ("Churn Threshold (T_c)",
                                [5, 10, 15, 20, 30, 50])),
    ("NUM_ISSUES",             ("Number of Issues (N)",
                                [1, 2, 3, 5])),
    ("TRUST_INFLUENCE",        ("Trust Influence (α)",
                                [0.0, 0.1, 0.2, 0.3, 0.5, 0.8])),
    ("BRIDGE_EFFICACY",        ("Bridge Efficacy",
                                [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])),
    ("BA_EDGE_PARAM",          ("BA Edge Parameter (m)",
                                [1, 2, 3, 5, 8])),
])

# KPIs to record at each sweep point
KPIS = ["Polarization", "Churn_Rate", "Revenue", "Avg_Frustration", "Active_Users"]


# ====================================================================
# Single run helper (designed to run in worker processes)
# ====================================================================

def _sweep_worker(args):
    """
    Worker function for multiprocessing.

    Each worker process gets its own copy of cfg (via fork/spawn),
    so patching cfg here is safe — it doesn't affect other workers.

    Args is a tuple:
        (param_name, param_value, enable_bridge, seed, run_id, label, max_steps)

    Returns a dict with all metadata + final KPIs.
    """
    param_name, param_value, enable_bridge, seed, run_id, label, max_steps = args

    # Restore defaults, then override the single OFAT parameter.
    # Pool workers are reused, so leftover setattr from prior jobs must be cleared.
    cfg.BETA_MEAN = 4.0
    cfg.ASSIMILATION_THRESHOLD = 0.3
    cfg.CHURN_THRESHOLD = 15
    cfg.NUM_ISSUES = 2
    cfg.TRUST_INFLUENCE = 0.3
    cfg.BRIDGE_EFFICACY = 0.46
    cfg.BA_EDGE_PARAM = 3
    cfg.ENABLE_TRUST = False
    cfg.ENABLE_BRIDGE = False

    setattr(cfg, param_name, param_value)
    if param_name == "TRUST_INFLUENCE":
        cfg.ENABLE_TRUST = True

    try:
        model = SocialNetworkModel(
            num_agents=cfg.NUM_AGENTS,
            ba_m=cfg.BA_EDGE_PARAM,
            enable_bridge=enable_bridge,
            enable_trust=cfg.ENABLE_TRUST,
            seed=seed,
        )
        for _ in range(max_steps):
            if not model.running:
                break
            model.step()

        # Extract final-step KPIs
        df = model.datacollector.get_model_vars_dataframe()
        final = df.iloc[-1]
        result = {
            "Parameter": param_name,
            "Label": label,
            "Value": param_value,
            "Bridge": enable_bridge,
            "Run": run_id,
        }
        result.update({kpi: final.get(kpi, np.nan) for kpi in KPIS})
        return result

    except Exception as e:
        print(f"  [ERROR] {param_name}={param_value}, bridge={enable_bridge}, "
              f"run={run_id}: {e}")
        return None


# ====================================================================
# OFAT sweep engine (parallelised)
# ====================================================================

def ofat_sweep(iterations=None, quick=False, n_workers=None):
    """
    One-Factor-At-a-Time sweep with multiprocessing.

    For each parameter, sweep across its defined range while holding
    all other parameters at their config defaults.  For each sweep
    point, run `iterations` simulations per condition.

    Parameters
    ----------
    iterations : int, default cfg.BATCH_ITERATIONS
        Number of independent runs per sweep point per condition.
    quick : bool
        If True, use 3 iterations instead of full batch (for testing).
    n_workers : int, default mp.cpu_count()
        Number of parallel worker processes.

    Returns
    -------
    pd.DataFrame with columns:
        [Parameter, Value, Bridge, Run, <KPIs...>]
    """
    if iterations is None:
        iterations = 3 if quick else cfg.BATCH_ITERATIONS
    if n_workers is None:
        # Cap default workers so the host UI stays usable (override with --workers=N).
        n_workers = max(1, min(2, mp.cpu_count()))

    base_seed = cfg.RANDOM_SEED
    max_steps = cfg.MAX_STEPS

    # Build the full job list upfront
    jobs = []
    for param_name, (label, values) in SWEEP_PARAMS.items():
        for val in values:
            for bridge in [False, True]:
                for run_id in range(iterations):
                    seed = base_seed + run_id
                    jobs.append((param_name, val, bridge, seed,
                                 run_id, label, max_steps))

    total_jobs = len(jobs)
    print(f"  Total jobs: {total_jobs}  ({n_workers} workers)", flush=True)

    # Execute in parallel
    all_rows = []
    completed = 0

    with mp.Pool(processes=n_workers) as pool:
        for result in pool.imap_unordered(_sweep_worker, jobs):
            if result is not None:
                all_rows.append(result)
            completed += 1
            if completed % 50 == 0 or completed == total_jobs:
                pct = 100 * completed / total_jobs
                print(f"  Progress: {completed:>5}/{total_jobs} ({pct:.0f}%)", flush=True)

    return pd.DataFrame(all_rows)


# ====================================================================
# Visualization
# ====================================================================

def plot_sensitivity(results_df):
    """
    Generate one figure per parameter showing how each KPI responds
    to the parameter sweep, comparing Bridge OFF vs ON.
    """
    COLORS = {"OFF": "#E74C3C", "ON": "#2980B9"}

    for param_name, (label, values) in SWEEP_PARAMS.items():
        subset = results_df[results_df["Parameter"] == param_name]
        if subset.empty:
            continue

        fig, axes = plt.subplots(1, len(KPIS), figsize=(4 * len(KPIS), 4))
        if len(KPIS) == 1:
            axes = [axes]

        for ax, kpi in zip(axes, KPIS):
            for bridge_val, color_key in [(False, "OFF"), (True, "ON")]:
                sub = subset[subset["Bridge"] == bridge_val]
                grouped = sub.groupby("Value")[kpi]
                means = grouped.mean()
                stds = grouped.std()

                ax.plot(means.index, means.values,
                        color=COLORS[color_key], marker='o', markersize=4,
                        linewidth=1.8, label=f"Bridge {color_key}")
                ax.fill_between(
                    means.index,
                    (means - stds).values,
                    (means + stds).values,
                    color=COLORS[color_key], alpha=0.12
                )

            ax.set_xlabel(label, fontsize=9)
            ax.set_ylabel(kpi.replace("_", " "), fontsize=9)
            ax.tick_params(labelsize=8)
            ax.legend(fontsize=7, loc="best")

        fig.suptitle(f"Sensitivity: {label}", fontweight="bold", fontsize=12)
        plt.tight_layout()
        fname = f"sensitivity_{param_name.lower()}.png"
        fig.savefig(os.path.join(FIG_DIR, fname), dpi=100, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {fname}")


def plot_sensitivity_heatmap(results_df):
    """
    Generate a summary heatmap: rows = parameters, columns = KPIs.
    Cell values = Cohen's d (Bridge ON vs OFF) at default parameter value.

    This gives a single-glance overview of which parameters and KPIs
    are most affected by the Bridge Algorithm.
    """
    def _cohens_d(g1, g2):
        """Inline Cohen's d to avoid stdlib statistics name collision."""
        g1, g2 = np.asarray(g1, dtype=float), np.asarray(g2, dtype=float)
        n1, n2 = len(g1), len(g2)
        if n1 < 2 or n2 < 2:
            return 0.0
        pooled = np.sqrt(((n1-1)*np.var(g1, ddof=1) + (n2-1)*np.var(g2, ddof=1)) / (n1+n2-2))
        return float((np.mean(g1) - np.mean(g2)) / pooled) if pooled > 1e-12 else 0.0

    rows = []
    for param_name, (label, values) in SWEEP_PARAMS.items():
        subset = results_df[results_df["Parameter"] == param_name]

        for kpi in KPIS:
            # Aggregate across all sweep values to get overall effect
            g_off = subset[subset["Bridge"] == False][kpi].dropna().values
            g_on = subset[subset["Bridge"] == True][kpi].dropna().values
            if len(g_off) >= 2 and len(g_on) >= 2:
                d = _cohens_d(g_off, g_on)
            else:
                d = 0.0
            rows.append({"Parameter": label, "KPI": kpi, "Cohen_d": d})

    heat_df = pd.DataFrame(rows).pivot(
        index="Parameter", columns="KPI", values="Cohen_d"
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(
        heat_df, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
        linewidths=0.5, ax=ax, cbar_kws={"label": "Cohen's d"}
    )
    ax.set_title("Sensitivity Overview: Cohen's d (Bridge ON vs OFF)",
                 fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "sensitivity_heatmap.png"),
                dpi=100, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: sensitivity_heatmap.png")


# ====================================================================
# Entry point
# ====================================================================

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    quick = "--quick" in sys.argv
    if quick:
        print("  Quick mode: 3 iterations per sweep point", flush=True)
    if "--laptop" in sys.argv:
        print("  Laptop mode: 2 workers, 5 iters (UI stays usable)", flush=True)

    wall_start = time.perf_counter()

    print("=" * 54, flush=True)
    print("  OFAT Sensitivity Analysis — Dynamic-BABE Model", flush=True)
    print("=" * 54, flush=True)

    # CLI: --iters=N  --workers=N  --quick  --laptop
    iterations = None
    n_workers = None
    for arg in sys.argv:
        if arg.startswith("--iters="):
            iterations = int(arg.split("=", 1)[1])
        if arg.startswith("--workers="):
            n_workers = int(arg.split("=", 1)[1])

    if "--laptop" in sys.argv:
        if iterations is None:
            iterations = 5
        if n_workers is None:
            n_workers = 2

    results = ofat_sweep(quick=quick, iterations=iterations, n_workers=n_workers)

    # Export raw results
    csv_path = os.path.join(OUT_DIR, "sensitivity_results.csv")
    results.to_csv(csv_path, index=False)
    print(f"\n  Raw results: {csv_path}", flush=True)

    # Generate figures
    print("\nGenerating sensitivity figures...", flush=True)
    plot_sensitivity(results)
    plot_sensitivity_heatmap(results)

    elapsed = time.perf_counter() - wall_start
    print(f"\n  Sensitivity analysis complete in {elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    main()

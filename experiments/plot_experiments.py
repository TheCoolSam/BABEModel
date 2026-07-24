"""Plot ablation / baseline / topology experiment summaries."""

from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(_ROOT, "figures")
OUT = os.path.join(_ROOT, "output")
os.makedirs(FIG, exist_ok=True)
sns.set_theme(style="whitegrid", font_scale=1.0)


def _save(fig, name: str):
    path = os.path.join(FIG, name)
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print("Saved", path)


def plot_ablations(path: str):
    df = pd.read_csv(path)
    # Focus paradox KPIs vs REF and M0
    order = [
        "REF_baseline",
        "M0_bridge_intercept",
        "M1_heal_only",
        "M2_bridge_drop",
        "M3_no_heal_bonus",
        "M4_full_no_decay",
        "M5_no_brand_safety",
    ]
    df = df[df["Condition"].isin(order)]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
    for ax, kpi in zip(axes, ["Polarization", "Churn_Rate", "Revenue"]):
        sns.barplot(data=df, x="Condition", y=kpi, ax=ax, errorbar="sd", color="#2980B9")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=7)
        ax.set_xlabel("")
        ax.set_title(kpi.replace("_", " "))
    fig.suptitle("Mechanism ablations (mean ± SD)", fontweight="bold")
    plt.tight_layout()
    _save(fig, "fig_ablations.png")


def plot_baselines(path: str):
    df = pd.read_csv(path)
    fig, ax = plt.subplots(figsize=(5, 3.5))
    sns.barplot(data=df, x="Model", y="Polarization", ax=ax, errorbar="sd", color="#27AE60")
    ax.set_title("Baseline model comparison: final polarization")
    plt.tight_layout()
    _save(fig, "fig_baselines_polarization.png")


def plot_topology(path: str):
    df = pd.read_csv(path)
    # Display-only rename: CSV column stays "Bridge"
    if "Bridge" in df.columns:
        df = df.copy()
        df["Filter"] = df["Bridge"].map({True: "Filter ON", False: "Filter OFF"})
        hue_col = "Filter"
    else:
        hue_col = "Bridge"
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
    for ax, kpi in zip(axes, ["Polarization", "Churn_Rate"]):
        sns.barplot(
            data=df,
            x="Topology",
            y=kpi,
            hue=hue_col,
            ax=ax,
            errorbar="sd",
        )
        ax.set_title(kpi.replace("_", " "))
        # Ensure legend uses Filter wording
        leg = ax.get_legend()
        if leg is not None:
            leg.set_title("Filter")
    fig.suptitle("Topology robustness (BA vs Watts–Strogatz)", fontweight="bold")
    plt.tight_layout()
    _save(fig, "fig_topology_robustness.png")


def main():
    abl = os.path.join(OUT, "ablation_results.csv")
    base = os.path.join(OUT, "baseline_compare.csv")
    topo = os.path.join(OUT, "topology_robustness.csv")
    if os.path.exists(abl):
        plot_ablations(abl)
    if os.path.exists(base):
        plot_baselines(base)
    if os.path.exists(topo):
        plot_topology(topo)


if __name__ == "__main__":
    main()

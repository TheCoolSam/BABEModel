"""
visualize.py — Publication-Quality Figures for Dynamic-BABE Model
==================================================================
Reads the batch_run.py CSV output (2×2 factorial design) and generates
presentation-ready PNG figures in the ./figures/ directory.

Run:
    python visualize.py

Conditions (2×2 factorial):
    Baseline     — Bridge OFF, Trust OFF
    Bridge Only  — Bridge ON,  Trust OFF
    Trust Only   — Bridge OFF, Trust ON
    Full Model   — Bridge ON,  Trust ON

Figures produced (main text):
    fig2_polarization_factorial.png  — Polarization over time, 4 conditions
    fig3_churn_factorial.png         — Churn rate over time, 4 conditions
    fig4_revenue_factorial.png       — Revenue over time, 4 conditions
    fig5_summary_barplot.png         — Grouped bar chart: 5 KPIs × 4 conditions
    fig6_trust_dynamics.png          — Dual panel: mean trust + ingroup/outgroup
    fig7_echo_chambers.png           — Opinion clustering coefficient, 4 conditions

Figures produced (supplementary):
    figS1_frustration_factorial.png  — Avg frustration, 4 conditions
    figS2_active_users_factorial.png — Active users, 4 conditions
    figS3_extremity_factorial.png    — Opinion extremity from agent data
    figS6_trust_segregation.png      — Trust segregation ratio, trust-enabled only
"""

import os
from collections import OrderedDict
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for PNG export
import matplotlib.pyplot as plt
import seaborn as sns

# Style
sns.set_theme(style="whitegrid", font_scale=1.2)

CONDITIONS = OrderedDict([
    ("Baseline",     {"color": "#E74C3C", "ls": "-"}),      # Red
    ("Bridge Only",  {"color": "#2980B9", "ls": "-"}),      # Blue
    ("Trust Only",   {"color": "#27AE60", "ls": "--"}),     # Green dashed
    ("Full Model",   {"color": "#8E44AD", "ls": "--"}),     # Purple dashed
])

# Map condition names to CSV filename suffixes
_CSV_MAP = OrderedDict([
    ("Baseline",    "baseline"),
    ("Bridge Only", "bridge_only"),
    ("Trust Only",  "trust_only"),
    ("Full Model",  "full_model"),
])

OUT_DIR = os.path.join(os.path.dirname(__file__), "figures")
DATA_DIR = os.path.join(os.path.dirname(__file__), "output")


def load_data():
    """Load and tag all 4 model-level CSVs, skipping any that are missing."""
    frames = []
    for condition, suffix in _CSV_MAP.items():
        path = os.path.join(DATA_DIR, "model_data_{}.csv".format(suffix))
        if not os.path.exists(path):
            print("  [skip] {} not found — skipping condition '{}'".format(
                os.path.basename(path), condition))
            continue
        tmp = pd.read_csv(path)
        tmp["Condition"] = condition
        frames.append(tmp)
    if not frames:
        raise FileNotFoundError("No model-level CSVs found in {}".format(DATA_DIR))
    return pd.concat(frames, ignore_index=True)


def load_agent_data():
    """Load and tag all 4 agent-level CSVs, skipping any that are missing."""
    frames = []
    for condition, suffix in _CSV_MAP.items():
        path = os.path.join(DATA_DIR, "agent_data_{}.csv".format(suffix))
        if not os.path.exists(path):
            print("  [skip] {} not found — skipping condition '{}'".format(
                os.path.basename(path), condition))
            continue
        tmp = pd.read_csv(path)
        tmp["Condition"] = condition
        frames.append(tmp)
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def plot_metric(df, metric, ylabel, title, filename,
                conditions=None):
    """
    Plot a single KPI over time with mean ± 95% CI shading,
    comparing the specified conditions (defaults to all 4).
    """
    if conditions is None:
        conditions = CONDITIONS

    fig, ax = plt.subplots(figsize=(10, 5))

    for condition, style in conditions.items():
        subset = df[df["Condition"] == condition]
        if subset.empty:
            continue
        grouped = subset.groupby("Step")[metric]
        mean = grouped.mean()
        std = grouped.std()
        n = grouped.count()
        ci = 1.96 * std / np.sqrt(n)

        ax.plot(mean.index, mean.values,
                color=style["color"], linestyle=style["ls"],
                linewidth=2, label=condition)
        ax.fill_between(mean.index,
                        (mean - ci).values,
                        (mean + ci).values,
                        color=style["color"], alpha=0.15)

    ax.set_xlabel("Simulation Step (t)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")
    ax.legend(frameon=True, loc="best")
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, filename), dpi=300)
    plt.close(fig)
    print("  Saved: {}".format(filename))


def plot_summary_bars(df):
    """
    Grouped bar chart comparing final-step means for 5 KPIs across
    all 4 factorial conditions, with error bars (± 1 SD).
    """
    # Get last step for each run
    final = df.groupby(["Run", "Condition"]).last().reset_index()

    metrics = ["Polarization", "Revenue", "Churn_Rate",
               "Avg_Frustration", "Active_Users"]
    labels = ["Polarization\n(sigma)", "Revenue\n($)",
              "Churn Rate\n(%)", "Avg Frustration\n(F_i)",
              "Active Users\n(count)"]

    fig, axes = plt.subplots(1, 5, figsize=(22, 5))

    cond_names = list(CONDITIONS.keys())
    n_conds = len(cond_names)
    bar_width = 0.18
    x_offsets = np.arange(n_conds) * bar_width - (n_conds - 1) * bar_width / 2

    for ax, metric, label in zip(axes, metrics, labels):
        for i, cond in enumerate(cond_names):
            data = final[final["Condition"] == cond][metric]
            m = data.mean()
            s = data.std()
            bar = ax.bar(x_offsets[i], m, width=bar_width,
                         yerr=s, capsize=4,
                         color=CONDITIONS[cond]["color"],
                         edgecolor="black", linewidth=0.5,
                         label=cond if metric == metrics[0] else "")
            # Annotate bar value above error-bar cap
            cap_top = m + s if not np.isnan(s) else m
            y_pos = cap_top + max(abs(cap_top) * 0.02, 0.005)
            ax.text(x_offsets[i], y_pos,
                    "{:.3f}".format(m), ha="center", va="bottom", fontsize=7)

        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.set_xticks([])
        # Auto-scale with headroom
        ax.autoscale(axis="y")
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin, ymax * 1.25 if ymax > 0 else 1.0)

    # Single shared legend at the bottom
    handles, lbls = axes[0].get_legend_handles_labels()
    fig.legend(handles, lbls, loc="lower center",
               ncol=n_conds, frameon=True, fontsize=10,
               bbox_to_anchor=(0.5, -0.02))

    n_runs = len(final[final["Condition"] == cond_names[0]]["Run"].unique()) \
        if "Run" in final.columns else "?"
    fig.suptitle(
        "Dynamic-BABE Model — Final-Step Comparison (2×2 Factorial, n={} runs/condition)".format(n_runs),
        fontweight="bold", fontsize=13, y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig5_summary_barplot.png"),
                dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: fig5_summary_barplot.png")


def plot_extremity(agent_df):
    """
    Plot mean Opinion Extremity over time from agent-level data.
    Extremity = ||O_i|| / sqrt(N) — captures multidimensional radicalism.
    """
    if agent_df is None or "Opinion_Extremity" not in agent_df.columns:
        print("  Skipped: figS3_extremity_factorial.png (no Opinion_Extremity data)")
        return

    fig, ax = plt.subplots(figsize=(10, 5))

    for condition, style in CONDITIONS.items():
        subset = agent_df[agent_df["Condition"] == condition]
        if subset.empty:
            continue
        # Mean extremity per step per run, then aggregate across runs
        per_step_run = subset.groupby(["Step", "Run"])["Opinion_Extremity"].mean().reset_index()
        grouped = per_step_run.groupby("Step")["Opinion_Extremity"]
        mean = grouped.mean()
        std = grouped.std()
        n = grouped.count()
        ci = 1.96 * std / np.sqrt(n)

        ax.plot(mean.index, mean.values,
                color=style["color"], linestyle=style["ls"],
                linewidth=2, label=condition)
        ax.fill_between(mean.index,
                        (mean - ci).values,
                        (mean + ci).values,
                        color=style["color"], alpha=0.15)

    ax.set_xlabel("Simulation Step (t)")
    ax.set_ylabel("Mean Opinion Extremity (||O|| / sqrt(N))")
    ax.set_title("Opinion Extremity Over Time — 2×2 Factorial", fontweight="bold")
    ax.legend(frameon=True, loc="best")
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "figS3_extremity_factorial.png"), dpi=300)
    plt.close(fig)
    print("  Saved: figS3_extremity_factorial.png")


# ====================================================================
# Trust-specific helpers
# ====================================================================

_TRUST_CONDITIONS = OrderedDict([
    ("Trust Only",  CONDITIONS["Trust Only"]),
    ("Full Model",  CONDITIONS["Full Model"]),
])


def plot_trust_dynamics(df):
    """
    Fig 6 — Dual panel:
      (A) Mean trust over time for trust-enabled conditions
      (B) Ingroup vs Outgroup trust divergence for trust-enabled conditions
    """
    if "Mean_Trust" not in df.columns:
        print("  Skipped: fig6_trust_dynamics.png (no trust data)")
        return

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(14, 5))

    # Panel A — Mean Trust
    for condition, style in _TRUST_CONDITIONS.items():
        subset = df[df["Condition"] == condition]
        if subset.empty:
            continue
        grouped = subset.groupby("Step")["Mean_Trust"]
        mean = grouped.mean()
        std = grouped.std()
        n = grouped.count()
        ci = 1.96 * std / np.sqrt(n)

        ax_a.plot(mean.index, mean.values,
                  color=style["color"], linestyle=style["ls"],
                  linewidth=2, label=condition)
        ax_a.fill_between(mean.index,
                          (mean - ci).values,
                          (mean + ci).values,
                          color=style["color"], alpha=0.15)

    ax_a.set_xlabel("Simulation Step (t)")
    ax_a.set_ylabel("Mean Dyadic Trust (T)")
    ax_a.set_title("(A) Network-Wide Mean Trust", fontweight="bold")
    ax_a.legend(frameon=True, loc="best")

    # Panel B — Ingroup vs Outgroup divergence
    if "Ingroup_Trust" in df.columns and "Outgroup_Trust" in df.columns:
        for condition, style in _TRUST_CONDITIONS.items():
            subset = df[df["Condition"] == condition]
            if subset.empty:
                continue
            for metric, line_label, ls in [
                ("Ingroup_Trust", "{} (ingroup)".format(condition), "-"),
                ("Outgroup_Trust", "{} (outgroup)".format(condition), ":"),
            ]:
                grouped = subset.groupby("Step")[metric]
                mean = grouped.mean()
                std = grouped.std()
                n = grouped.count()
                ci = 1.96 * std / np.sqrt(n)

                ax_b.plot(mean.index, mean.values,
                          color=style["color"], linestyle=ls,
                          linewidth=2, label=line_label)
                ax_b.fill_between(mean.index,
                                  (mean - ci).values,
                                  (mean + ci).values,
                                  color=style["color"], alpha=0.1)

    ax_b.set_xlabel("Simulation Step (t)")
    ax_b.set_ylabel("Mean Trust (T)")
    ax_b.set_title("(B) Ingroup vs Outgroup Trust", fontweight="bold")
    ax_b.legend(frameon=True, loc="best", fontsize=9)

    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig6_trust_dynamics.png"), dpi=300)
    plt.close(fig)
    print("  Saved: fig6_trust_dynamics.png")


def plot_echo_chamber(df):
    """
    Fig 7 — Opinion Clustering Coefficient over time, all 4 conditions.
    """
    if "Opinion_Clustering" not in df.columns:
        print("  Skipped: fig7_echo_chambers.png (no clustering data)")
        return

    plot_metric(df, "Opinion_Clustering",
                "Opinion Clustering Coefficient",
                "Echo Chamber Formation — 2×2 Factorial",
                "fig7_echo_chambers.png")


def plot_trust_segregation(df):
    """
    FigS6 — Trust Segregation ratio over time for trust-enabled conditions.
    """
    if "Trust_Segregation" not in df.columns:
        print("  Skipped: figS6_trust_segregation.png (no Trust_Segregation data)")
        return

    plot_metric(df, "Trust_Segregation",
                "Trust Segregation Ratio",
                "Trust Segregation — Trust-Enabled Conditions",
                "figS6_trust_segregation.png",
                conditions=_TRUST_CONDITIONS)


# ====================================================================
# Main
# ====================================================================

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = load_data()
    agent_df = load_agent_data()

    print("Generating publication figures (2×2 factorial)...")
    print()

    # ---- Main text figures ----
    plot_metric(df, "Polarization", "Global Polarization (sigma)",
                "Polarization Over Time — 2×2 Factorial",
                "fig2_polarization_factorial.png")

    plot_metric(df, "Churn_Rate", "Churn Rate (fraction)",
                "Cumulative Churn Rate — 2×2 Factorial",
                "fig3_churn_factorial.png")

    plot_metric(df, "Revenue", "Platform Revenue ($)",
                "Revenue Over Time — 2×2 Factorial",
                "fig4_revenue_factorial.png")

    plot_summary_bars(df)

    plot_trust_dynamics(df)

    plot_echo_chamber(df)

    # ---- Supplementary figures ----
    plot_metric(df, "Avg_Frustration", "Average Frustration (F_i)",
                "Mean Agent Frustration — 2×2 Factorial",
                "figS1_frustration_factorial.png")

    plot_metric(df, "Active_Users", "Active Users (count)",
                "User Retention — 2×2 Factorial",
                "figS2_active_users_factorial.png")

    plot_extremity(agent_df)

    plot_trust_segregation(df)

    print()
    print("All figures saved to: {}".format(OUT_DIR))


if __name__ == "__main__":
    main()

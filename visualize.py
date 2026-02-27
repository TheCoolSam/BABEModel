"""
visualize.py — Publication-Quality Figures for BSSEF Presentation
==================================================================
Reads the batch_run.py CSV output and generates 5 presentation-ready
PNG figures in the ./figures/ directory.

Run:
    python visualize.py

Figures produced:
    1. polarization_over_time.png   — Bridge OFF vs ON  (with CI band)
    2. revenue_over_time.png        — Revenue collapse curve
    3. churn_rate_over_time.png     — Cumulative churn trajectory
    4. frustration_over_time.png    — Average frustration build-up
    5. active_users_over_time.png   — User retention comparison
    6. summary_barplot.png          — Final-step side-by-side comparison
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for PNG export
import matplotlib.pyplot as plt
import seaborn as sns

# ── Style ─────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", font_scale=1.2)
COLORS = {"Bridge OFF": "#E74C3C", "Bridge ON": "#2980B9"}

OUT_DIR = os.path.join(os.path.dirname(__file__), "figures")
DATA_DIR = os.path.join(os.path.dirname(__file__), "output")


def load_data():
    """Load and tag the two model-level CSVs."""
    off = pd.read_csv(os.path.join(DATA_DIR, "model_data_bridge_off.csv"))
    on = pd.read_csv(os.path.join(DATA_DIR, "model_data_bridge_on.csv"))
    off["Condition"] = "Bridge OFF"
    on["Condition"] = "Bridge ON"
    return pd.concat([off, on], ignore_index=True)


def plot_metric(df, metric, ylabel, title, filename):
    """
    Plot a single KPI over time with mean ± 95% CI shading,
    comparing Bridge OFF vs Bridge ON.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    for condition, color in COLORS.items():
        subset = df[df["Condition"] == condition]
        grouped = subset.groupby("Step")[metric]
        mean = grouped.mean()
        std = grouped.std()
        n = grouped.count()
        ci = 1.96 * std / np.sqrt(n)

        ax.plot(mean.index, mean.values, color=color, linewidth=2,
                label=condition)
        ax.fill_between(mean.index,
                        (mean - ci).values,
                        (mean + ci).values,
                        color=color, alpha=0.15)

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
    Side-by-side bar chart comparing final-step means for all KPIs.
    """
    # Get last step for each run
    final = df.groupby(["Run", "Condition"]).last().reset_index()

    metrics = ["Polarization", "Revenue", "Churn_Rate",
               "Avg_Frustration", "Active_Users"]
    labels = ["Polarization\n(sigma)", "Revenue\n($)",
              "Churn Rate\n(%)", "Avg Frustration\n(F_i)",
              "Active Users\n(count)"]

    fig, axes = plt.subplots(1, 5, figsize=(18, 4.5))

    for ax, metric, label in zip(axes, metrics, labels):
        data_off = final[final["Condition"] == "Bridge OFF"][metric]
        data_on = final[final["Condition"] == "Bridge ON"][metric]

        means = [data_off.mean(), data_on.mean()]
        stds = [data_off.std(), data_on.std()]

        bars = ax.bar(["Bridge\nOFF", "Bridge\nON"], means,
                      yerr=stds, capsize=5,
                      color=[COLORS["Bridge OFF"], COLORS["Bridge ON"]],
                      edgecolor="black", linewidth=0.5)
        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.tick_params(axis="x", labelsize=9)

        # Annotate bar values — place text ABOVE the error bar cap
        for bar, m, s in zip(bars, means, stds):
            cap_top = bar.get_height() + s  # top of error bar
            y_pos = cap_top + max(cap_top * 0.02, 0.005)  # small gap above cap
            ax.text(bar.get_x() + bar.get_width() / 2, y_pos,
                    "{:.3f}".format(m), ha="center", va="bottom", fontsize=8)
        # Auto-scale y-axis with enough headroom for labels
        ax.autoscale(axis="y")
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin, ymax * 1.20 if ymax > 0 else 1.0)

    fig.suptitle("Dynamic-BABE Model — Final-Step Comparison (n={} runs per condition)".format(
                     len(final['Run'].unique()) // 2 if 'Run' in final.columns else '?'),
                 fontweight="bold", fontsize=13, y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "summary_barplot.png"),
                dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: summary_barplot.png")


def load_agent_data():
    """Load and tag the two agent-level CSVs."""
    off_path = os.path.join(DATA_DIR, "agent_data_bridge_off.csv")
    on_path = os.path.join(DATA_DIR, "agent_data_bridge_on.csv")
    if not (os.path.exists(off_path) and os.path.exists(on_path)):
        return None
    off = pd.read_csv(off_path)
    on = pd.read_csv(on_path)
    off["Condition"] = "Bridge OFF"
    on["Condition"] = "Bridge ON"
    return pd.concat([off, on], ignore_index=True)


def plot_extremity(agent_df):
    """
    Plot mean Opinion Extremity over time from agent-level data.
    Extremity = ||O_i|| / sqrt(N) — captures multidimensional radicalism.
    """
    if agent_df is None or "Opinion_Extremity" not in agent_df.columns:
        print("  Skipped: extremity_over_time.png (no Opinion_Extremity data)")
        return

    fig, ax = plt.subplots(figsize=(10, 5))

    for condition, color in COLORS.items():
        subset = agent_df[agent_df["Condition"] == condition]
        # Mean extremity per step per run, then aggregate across runs
        per_step_run = subset.groupby(["Step", "Run"])["Opinion_Extremity"].mean().reset_index()
        grouped = per_step_run.groupby("Step")["Opinion_Extremity"]
        mean = grouped.mean()
        std = grouped.std()
        n = grouped.count()
        ci = 1.96 * std / np.sqrt(n)

        ax.plot(mean.index, mean.values, color=color, linewidth=2,
                label=condition)
        ax.fill_between(mean.index,
                        (mean - ci).values,
                        (mean + ci).values,
                        color=color, alpha=0.15)

    ax.set_xlabel("Simulation Step (t)")
    ax.set_ylabel("Mean Opinion Extremity (||O|| / sqrt(N))")
    ax.set_title("Opinion Extremity Over Time — Bridge OFF vs ON", fontweight="bold")
    ax.legend(frameon=True, loc="best")
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "extremity_over_time.png"), dpi=300)
    plt.close(fig)
    print("  Saved: extremity_over_time.png")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = load_data()
    agent_df = load_agent_data()

    print("Generating BSSEF presentation figures...")
    print()

    plot_metric(df, "Polarization", "Global Polarization (sigma)",
                "Polarization Over Time — Bridge OFF vs ON",
                "polarization_over_time.png")

    plot_metric(df, "Revenue", "Platform Revenue ($)",
                "Revenue Over Time — Bridge OFF vs ON",
                "revenue_over_time.png")

    plot_metric(df, "Churn_Rate", "Churn Rate (fraction)",
                "Cumulative Churn Rate — Bridge OFF vs ON",
                "churn_rate_over_time.png")

    plot_metric(df, "Avg_Frustration", "Average Frustration (F_i)",
                "Mean Agent Frustration — Bridge OFF vs ON",
                "frustration_over_time.png")

    plot_metric(df, "Active_Users", "Active Users (count)",
                "User Retention — Bridge OFF vs ON",
                "active_users_over_time.png")

    plot_extremity(agent_df)

    plot_summary_bars(df)

    print()
    print("All figures saved to: {}".format(OUT_DIR))
    print("Ready for BSSEF presentation!")


if __name__ == "__main__":
    main()

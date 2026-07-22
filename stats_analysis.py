"""
stats_analysis.py — Publication-Quality Statistical Analysis for Dynamic-BABE
=========================================================================
Reads batch_run.py 2×2 factorial CSV outputs and produces:

  1. Pairwise Mann-Whitney U tests (non-parametric, no normality assumption)
  2. Cohen's d effect sizes (standardised magnitude of difference)
  3. Bootstrap 95% Confidence Intervals for each condition
  4. Bridge × Trust Interaction Analysis (Wilcoxon signed-rank on seed-paired contrasts)
  5. Formatted output suitable for publication tables and supplementary material

Pairwise Comparisons (6 pairs):
    - Baseline vs Bridge Only
    - Baseline vs Trust Only
    - Baseline vs Full Model
    - Bridge Only vs Trust Only
    - Bridge Only vs Full Model
    - Trust Only vs Full Model

Run:
    python stats_analysis.py
"""

import os
from collections import OrderedDict
import numpy as np
import pandas as pd
from scipy import stats

DATA_DIR = os.path.join(os.path.dirname(__file__), "output")
OUT_DIR = os.path.join(os.path.dirname(__file__), "output")

# ====================================================================
# Core statistical functions
# ====================================================================

def cohens_d(group1, group2):
    """
    Compute Cohen's d — the standardised mean difference.
    """
    g1 = np.asarray(group1, dtype=float)
    g2 = np.asarray(group2, dtype=float)
    n1, n2 = len(g1), len(g2)
    if n1 < 2 or n2 < 2:
        return 0.0
    var1 = np.var(g1, ddof=1)
    var2 = np.var(g2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std < 1e-12:
        return 0.0
    return float((np.mean(g1) - np.mean(g2)) / pooled_std)


def d_magnitude(d):
    """Human-readable effect-size label (Cohen, 1988)."""
    ad = abs(d)
    if ad < 0.2:
        return "negligible"
    elif ad < 0.5:
        return "small"
    elif ad < 0.8:
        return "medium"
    else:
        return "large"


def mann_whitney_u(group1, group2):
    """
    Two-sided Mann-Whitney U test (non-parametric).
    """
    g1 = np.asarray(group1, dtype=float)
    g2 = np.asarray(group2, dtype=float)
    if len(g1) < 2 or len(g2) < 2:
        return 0.0, 1.0
    # If both groups have identical zero-variance data, return no difference
    if np.all(g1 == g1[0]) and np.all(g2 == g2[0]) and g1[0] == g2[0]:
        return 0.0, 1.0
    u_stat, p_val = stats.mannwhitneyu(g1, g2, alternative='two-sided')
    return float(u_stat), float(p_val)


def bootstrap_ci(data, statistic=np.mean, n_boot=10000, ci=0.95, rng=None):
    """
    Non-parametric bootstrap confidence interval.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    data = np.asarray(data, dtype=float)
    if len(data) == 0:
        return np.nan, np.nan
    if np.all(data == data[0]):
        return float(data[0]), float(data[0])
    boot_stats = np.array([
        statistic(rng.choice(data, size=len(data), replace=True))
        for _ in range(n_boot)
    ])
    alpha = (1 - ci) / 2
    return float(np.percentile(boot_stats, 100 * alpha)), \
           float(np.percentile(boot_stats, 100 * (1 - alpha)))


def bonferroni_correct(p_values):
    """Apply Bonferroni correction to a list of p-values."""
    m = len(p_values)
    return [min(1.0, p * m) for p in p_values]


# ====================================================================
# Main analysis pipeline
# ====================================================================

def load_final_step_data():
    """
    Load all 4 model-level CSVs and extract the FINAL step of each run.
    Returns a dict of final-step DataFrames keyed by condition name.
    """
    csv_map = {
        "Baseline": "model_data_baseline.csv",
        "Bridge Only": "model_data_bridge_only.csv",
        "Trust Only": "model_data_trust_only.csv",
        "Full Model": "model_data_full_model.csv",
    }
    
    data_dict = {}
    for name, filename in csv_map.items():
        path = os.path.join(DATA_DIR, filename)
        if not os.path.exists(path):
            print(f"  [WARNING] File not found: {path} — skipping condition '{name}'")
            continue
        df = pd.read_csv(path)
        final_df = df.groupby("Run").last().reset_index()
        data_dict[name] = final_df
        
    return data_dict


def run_analysis(data_dict, metrics=None):
    """
    Perform pairwise comparisons across all available conditions.
    """
    conditions = list(data_dict.keys())
    if len(conditions) < 2:
        print("  [ERROR] At least 2 conditions must be loaded for analysis.")
        return pd.DataFrame()

    # Find common numeric metrics
    common_cols = None
    for name, df in data_dict.items():
        numeric_cols = set(df.select_dtypes(include=[np.number]).columns)
        if common_cols is None:
            common_cols = numeric_cols
        else:
            common_cols &= numeric_cols
            
    exclude = {"Run", "Step", "Bridge", "Trust"}
    all_metrics = sorted([c for c in common_cols if c not in exclude])
    
    if metrics is None:
        metrics = all_metrics
    else:
        metrics = [m for m in metrics if m in all_metrics]

    # Generate pairwise comparison list (6 comparisons for 4 conditions)
    pairs = []
    for i in range(len(conditions)):
        for j in range(i + 1, len(conditions)):
            pairs.append((conditions[i], conditions[j]))

    rows = []
    raw_p_values = []

    for c1, c2 in pairs:
        df1 = data_dict[c1]
        df2 = data_dict[c2]
        
        for metric in metrics:
            g1 = df1[metric].dropna().values
            g2 = df2[metric].dropna().values
            
            if len(g1) < 2 or len(g2) < 2:
                continue

            # Check if there is variation in either group (to avoid redundant identical-group comparisons)
            # e.g., Mean_Trust or Trust_Segregation in Baseline vs Bridge Only is identically flat.
            # We still record it, but highlight that there is no variation.
            no_var = np.all(g1 == g1[0]) and np.all(g2 == g2[0]) and g1[0] == g2[0]

            # Descriptive statistics
            m1, m2 = np.mean(g1), np.mean(g2)
            s1, s2 = np.std(g1, ddof=1), np.std(g2, ddof=1)

            # 95% Bootstrap CI
            ci1 = bootstrap_ci(g1)
            ci2 = bootstrap_ci(g2)

            # Effect size & Mann-Whitney U test
            d = cohens_d(g1, g2)
            u_stat, p_val = mann_whitney_u(g1, g2)
            
            raw_p_values.append(p_val)

            rows.append({
                "Comparison": f"{c1} vs {c2}",
                "Group1": c1,
                "Group2": c2,
                "Metric": metric,
                "N1": len(g1),
                "Mean1": m1,
                "SD1": s1,
                "CI95_1": f"[{ci1[0]:.4f}, {ci1[1]:.4f}]",
                "N2": len(g2),
                "Mean2": m2,
                "SD2": s2,
                "CI95_2": f"[{ci2[0]:.4f}, {ci2[1]:.4f}]",
                "Cohen_d": d,
                "Effect_Size": "none" if no_var else d_magnitude(d),
                "U_statistic": u_stat,
                "p_value_raw": p_val,
            })

    # Apply Bonferroni correction across ALL pairwise tests
    adjusted_p_values = bonferroni_correct(raw_p_values)
    for i, row in enumerate(rows):
        row["p_value_adjusted"] = adjusted_p_values[i]
        row["Significant_0.05"] = "Yes" if adjusted_p_values[i] < 0.05 else "No"

    return pd.DataFrame(rows)


def run_interaction_analysis(data_dict, metrics=None):
    """
    Test for Bridge × Trust interaction significance.
    Since simulations are paired by seed/Run ID, we can compute paired differences:
      effect_bridge_alone = (Bridge Only) - (Baseline)
      effect_bridge_with_trust = (Full Model) - (Trust Only)

    We test the paired contrast (effect_with_trust - effect_alone) with a
    Wilcoxon signed-rank test. If significant, the Bridge effect changes
    in the presence of Trust (moderation / interaction).
    """
    required = {"Baseline", "Bridge Only", "Trust Only", "Full Model"}
    if not required.issubset(data_dict.keys()):
        print("  [WARNING] Interaction analysis requires all 4 conditions.")
        return pd.DataFrame()

    # Align DataFrames by Run ID to ensure seed pairing
    df_baseline = data_dict["Baseline"].sort_values("Run").set_index("Run")
    df_bridge = data_dict["Bridge Only"].sort_values("Run").set_index("Run")
    df_trust = data_dict["Trust Only"].sort_values("Run").set_index("Run")
    df_full = data_dict["Full Model"].sort_values("Run").set_index("Run")

    # Find common indices
    common_runs = df_baseline.index.intersection(df_bridge.index)
    common_runs = common_runs.intersection(df_trust.index)
    common_runs = common_runs.intersection(df_full.index)

    df_baseline = df_baseline.loc[common_runs]
    df_bridge = df_bridge.loc[common_runs]
    df_trust = df_trust.loc[common_runs]
    df_full = df_full.loc[common_runs]

    if metrics is None:
        exclude = {"Step", "Bridge", "Trust"}
        metrics = sorted([c for c in df_baseline.columns if c not in exclude and pd.api.types.is_numeric_dtype(df_baseline[c])])

    interaction_rows = []
    
    for metric in metrics:
        # Paired differences aligned by Run
        diff_alone = df_bridge[metric] - df_baseline[metric]
        diff_with_trust = df_full[metric] - df_trust[metric]
        paired_contrast = (diff_with_trust - diff_alone).astype(float).values

        # Skip if contrast is identically zero
        if np.allclose(paired_contrast, 0.0):
            continue

        mean_alone = float(np.mean(diff_alone))
        mean_with_trust = float(np.mean(diff_with_trust))

        # Wilcoxon signed-rank on paired contrasts (omit zero differences)
        nonzero = paired_contrast[np.abs(paired_contrast) > 1e-15]
        if len(nonzero) < 2:
            w_stat, p_val = 0.0, 1.0
        else:
            try:
                w_stat, p_val = stats.wilcoxon(
                    nonzero, alternative="two-sided", zero_method="wilcox"
                )
                w_stat, p_val = float(w_stat), float(p_val)
            except ValueError:
                w_stat, p_val = 0.0, 1.0

        # Cohen's d of the bridge effects (same definition as before)
        pooled_std = np.sqrt((np.var(diff_alone, ddof=1) + np.var(diff_with_trust, ddof=1)) / 2.0)
        d = (mean_with_trust - mean_alone) / pooled_std if pooled_std > 1e-12 else 0.0

        interaction_rows.append({
            "Metric": metric,
            "Mean_Bridge_Effect_Alone": mean_alone,
            "Mean_Bridge_Effect_With_Trust": mean_with_trust,
            "Cohen_d_Interaction": d,
            "Effect_Magnitude": d_magnitude(d),
            "W_statistic": w_stat,
            "p_value_raw": p_val,
        })

    results_df = pd.DataFrame(interaction_rows)
    if not results_df.empty:
        # Apply Bonferroni correction to interaction tests
        adjusted = bonferroni_correct(results_df["p_value_raw"].tolist())
        results_df["p_value_adjusted"] = adjusted
        results_df["Significant_0.05"] = ["Yes" if p < 0.05 else "No" for p in adjusted]

    return results_df


def format_table(pairwise_df, interaction_df):
    """Print beautifully formatted tables to the terminal."""
    print()
    print("=" * 110)
    print("  DYNAMIC-BABE MODEL — PAIRWISE COMPARISONS (2×2 FACTORIAL DESIGN)")
    print("=" * 110)
    
    comparisons = pairwise_df["Comparison"].unique()
    for comp in comparisons:
        print(f"\n>>> Comparison: {comp}")
        print("-" * 110)
        subset = pairwise_df[pairwise_df["Comparison"] == comp]
        for _, row in subset.iterrows():
            # Skip metrics with zero variation in both groups
            if row["Effect_Size"] == "none":
                continue
            sig_mark = "  [** SIG **]" if row["Significant_0.05"] == "Yes" else ""
            print(f"  {row['Metric']:<22} | "
                  f"Mean1={row['Mean1']:.4f} vs Mean2={row['Mean2']:.4f} | "
                  f"d={row['Cohen_d']:+.4f} ({row['Effect_Size']:<10}) | "
                  f"adj_p={row['p_value_adjusted']:.6f}{sig_mark}")
            
    print("\n" + "=" * 110)
    print("  DYNAMIC-BABE MODEL — BRIDGE × TRUST INTERACTION / MODERATION ANALYSIS")
    print("  (Compares the Bridge Effect without Trust vs the Bridge Effect with Trust)")
    print("=" * 110)
    print(f"  {'Metric':<22} | {'Bridge Effect Alone':<22} | {'Bridge Effect w/Trust':<22} | {'Interaction d':<14} | {'W':<10} | {'adj_p':<8} | {'Significant':<11}")
    print("-" * 120)
    
    for _, row in interaction_df.iterrows():
        sig_mark = "Yes [**]" if row["Significant_0.05"] == "Yes" else "No"
        print(f"  {row['Metric']:<22} | "
              f"{row['Mean_Bridge_Effect_Alone']:+21.4f} | "
              f"{row['Mean_Bridge_Effect_With_Trust']:+21.4f} | "
              f"{row['Cohen_d_Interaction']:+13.4f} | "
              f"{row['W_statistic']:<10.1f} | "
              f"{row['p_value_adjusted']:.6f} | "
              f"{sig_mark:<11}")
    print("=" * 120)
    print("  * Pairwise: Mann-Whitney U. Interaction: Wilcoxon signed-rank on paired contrasts.")
    print("  * Bonferroni correction applied independently to pairwise tests and interaction tests.")
    print("=" * 120)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading final-step data from 2×2 factorial batch runs...")
    data_dict = load_final_step_data()

    if not data_dict:
        print("  [ERROR] No data could be loaded. Please run batch_run.py first.")
        return

    print("Running pairwise comparisons (6 pairs)...")
    pairwise_results = run_analysis(data_dict)

    print("Running Bridge × Trust interaction analysis...")
    interaction_results = run_interaction_analysis(data_dict)

    # Format and print the tables
    format_table(pairwise_results, interaction_results)

    # Save to CSV
    pairwise_results.to_csv(os.path.join(OUT_DIR, "statistical_analysis.csv"), index=False)
    interaction_results.to_csv(os.path.join(OUT_DIR, "interaction_analysis.csv"), index=False)
    
    print()
    print(f"  Pairwise results exported to: {os.path.join(OUT_DIR, 'statistical_analysis.csv')}")
    print(f"  Interaction results exported to: {os.path.join(OUT_DIR, 'interaction_analysis.csv')}")


if __name__ == "__main__":
    main()

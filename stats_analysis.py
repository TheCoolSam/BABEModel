"""
stats_analysis.py — Publication-Quality Statistical Analysis for Dynamic-BABE
=========================================================================
Reads batch_run.py 2×2 factorial CSV outputs and produces:

  1. Seed-paired Wilcoxon signed-rank tests (primary; matches paired design)
  2. Mann-Whitney U (supplementary unpaired sensitivity)
  3. Cohen's d (Group1 - Group2) as a parametric complement
  4. Bootstrap 95% CIs
  5. Filter × Trust interaction (Wilcoxon on seed-paired contrasts)

CSV condition names remain Baseline / Bridge Only / Trust Only / Full Model;
printed labels map Bridge Only → Filter Only.

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

_DISPLAY_ALIAS = {
    "Bridge Only": "Filter Only",
    "Bridge": "Filter",
}

# Trust KPIs are fixed when Trust OFF — Filter×Trust "interactions" on these
# are not comparable to retention moderation (Δ_alone ≡ 0 by design).
TRUST_KPIS = {
    "Mean_Trust",
    "Ingroup_Trust",
    "Outgroup_Trust",
    "Trust_Segregation",
}


def _display(name: str) -> str:
    return _DISPLAY_ALIAS.get(name, name)


def cohens_d(group1, group2):
    """Cohen's d = (mean(group1) - mean(group2)) / pooled_sd."""
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
    g1 = np.asarray(group1, dtype=float)
    g2 = np.asarray(group2, dtype=float)
    if len(g1) < 2 or len(g2) < 2:
        return 0.0, 1.0
    if np.all(g1 == g1[0]) and np.all(g2 == g2[0]) and g1[0] == g2[0]:
        return 0.0, 1.0
    u_stat, p_val = stats.mannwhitneyu(g1, g2, alternative="two-sided")
    return float(u_stat), float(p_val)


def wilcoxon_paired(diff):
    """Two-sided Wilcoxon signed-rank on paired differences (drop exact zeros)."""
    diff = np.asarray(diff, dtype=float)
    nonzero = diff[np.abs(diff) > 1e-15]
    if len(nonzero) < 2:
        return 0.0, 1.0
    if np.allclose(nonzero, 0.0):
        return 0.0, 1.0
    try:
        w_stat, p_val = stats.wilcoxon(
            nonzero, alternative="two-sided", zero_method="wilcox"
        )
        return float(w_stat), float(p_val)
    except ValueError:
        return 0.0, 1.0


def bootstrap_ci(data, statistic=np.mean, n_boot=10000, ci=0.95, rng=None):
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
    return float(np.percentile(boot_stats, 100 * alpha)), float(
        np.percentile(boot_stats, 100 * (1 - alpha))
    )


def bonferroni_correct(p_values):
    m = len(p_values)
    if m == 0:
        return []
    return [min(1.0, p * m) for p in p_values]


TRUST_SEGREGATION_CAP = 1e6


def _cap_trust_segregation(df: pd.DataFrame) -> pd.DataFrame:
    if "Trust_Segregation" not in df.columns:
        return df
    out = df.copy()
    col = out["Trust_Segregation"].astype(float)
    col = col.replace([np.inf, -np.inf], TRUST_SEGREGATION_CAP)
    col = col.clip(upper=TRUST_SEGREGATION_CAP)
    out["Trust_Segregation"] = col
    return out


def load_final_step_data():
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
        data_dict[name] = _cap_trust_segregation(final_df)

    return data_dict


def run_analysis(data_dict, metrics=None):
    """
    Pairwise comparisons with seed-paired Wilcoxon as primary inference.
    Degenerate identical flat contrasts are recorded but excluded from Bonferroni m.
    """
    conditions = list(data_dict.keys())
    if len(conditions) < 2:
        print("  [ERROR] At least 2 conditions must be loaded for analysis.")
        return pd.DataFrame()

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

    pairs = []
    for i in range(len(conditions)):
        for j in range(i + 1, len(conditions)):
            pairs.append((conditions[i], conditions[j]))

    rows = []
    family_p_values = []  # only non-degenerate tests enter Bonferroni

    for c1, c2 in pairs:
        df1 = data_dict[c1].sort_values("Run").set_index("Run")
        df2 = data_dict[c2].sort_values("Run").set_index("Run")
        common_runs = df1.index.intersection(df2.index)
        df1 = df1.loc[common_runs]
        df2 = df2.loc[common_runs]

        for metric in metrics:
            g1 = df1[metric].astype(float).values
            g2 = df2[metric].astype(float).values

            if len(g1) < 2 or len(g2) < 2:
                continue

            no_var = (
                np.all(g1 == g1[0])
                and np.all(g2 == g2[0])
                and g1[0] == g2[0]
            )

            m1, m2 = float(np.mean(g1)), float(np.mean(g2))
            s1 = float(np.std(g1, ddof=1)) if len(g1) > 1 else 0.0
            s2 = float(np.std(g2, ddof=1)) if len(g2) > 1 else 0.0
            ci1 = bootstrap_ci(g1)
            ci2 = bootstrap_ci(g2)

            d = cohens_d(g1, g2)
            u_stat, p_mwu = mann_whitney_u(g1, g2)
            diff = g1 - g2
            w_stat, p_w = wilcoxon_paired(diff)

            # Primary p: paired Wilcoxon; degenerate → p=1, excluded from family
            p_primary = 1.0 if no_var else p_w
            in_family = not no_var

            row = {
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
                "Mean_Diff_G1_minus_G2": m1 - m2,
                "Cohen_d": d,
                "Effect_Size": "none" if no_var else d_magnitude(d),
                "W_statistic": w_stat,
                "U_statistic": u_stat,
                "p_value_wilcoxon_raw": p_primary if not no_var else 1.0,
                "p_value_mwu_raw": p_mwu,
                "p_value_raw": p_primary,  # primary = Wilcoxon
                "Degenerate": "Yes" if no_var else "No",
                "In_Bonferroni_Family": "Yes" if in_family else "No",
                "Trust_KPI": "Yes" if metric in TRUST_KPIS else "No",
            }
            rows.append(row)
            if in_family:
                family_p_values.append((len(rows) - 1, p_primary))

    adjusted = bonferroni_correct([p for _, p in family_p_values])
    adj_map = {idx: adj for (idx, _), adj in zip(family_p_values, adjusted)}
    m_family = len(family_p_values)

    for i, row in enumerate(rows):
        if i in adj_map:
            row["p_value_adjusted"] = adj_map[i]
            row["Bonferroni_m"] = m_family
        else:
            row["p_value_adjusted"] = 1.0
            row["Bonferroni_m"] = m_family
        row["Significant_0.05"] = (
            "Yes" if row["p_value_adjusted"] < 0.05 else "No"
        )

    return pd.DataFrame(rows)


def run_interaction_analysis(data_dict, metrics=None):
    required = {"Baseline", "Bridge Only", "Trust Only", "Full Model"}
    if not required.issubset(data_dict.keys()):
        print("  [WARNING] Interaction analysis requires all 4 conditions.")
        return pd.DataFrame()

    df_baseline = data_dict["Baseline"].sort_values("Run").set_index("Run")
    df_bridge = data_dict["Bridge Only"].sort_values("Run").set_index("Run")
    df_trust = data_dict["Trust Only"].sort_values("Run").set_index("Run")
    df_full = data_dict["Full Model"].sort_values("Run").set_index("Run")

    common_runs = df_baseline.index.intersection(df_bridge.index)
    common_runs = common_runs.intersection(df_trust.index)
    common_runs = common_runs.intersection(df_full.index)

    df_baseline = df_baseline.loc[common_runs]
    df_bridge = df_bridge.loc[common_runs]
    df_trust = df_trust.loc[common_runs]
    df_full = df_full.loc[common_runs]

    if metrics is None:
        exclude = {"Step", "Bridge", "Trust"}
        metrics = sorted([
            c
            for c in df_baseline.columns
            if c not in exclude and pd.api.types.is_numeric_dtype(df_baseline[c])
        ])

    interaction_rows = []

    for metric in metrics:
        diff_alone = df_bridge[metric] - df_baseline[metric]
        diff_with_trust = df_full[metric] - df_trust[metric]
        paired_contrast = (diff_with_trust - diff_alone).astype(float).values

        if np.allclose(paired_contrast, 0.0):
            continue

        mean_alone = float(np.mean(diff_alone))
        mean_with_trust = float(np.mean(diff_with_trust))
        alone_is_structural_zero = bool(np.allclose(diff_alone.astype(float), 0.0))

        w_stat, p_val = wilcoxon_paired(paired_contrast)

        pooled_std = np.sqrt(
            (np.var(diff_alone, ddof=1) + np.var(diff_with_trust, ddof=1)) / 2.0
        )
        d = (
            (mean_with_trust - mean_alone) / pooled_std
            if pooled_std > 1e-12
            else 0.0
        )

        note = ""
        if metric in TRUST_KPIS and alone_is_structural_zero:
            note = (
                "Trust OFF makes Delta_alone=0 by design; "
                "not comparable to retention moderation"
            )

        interaction_rows.append({
            "Metric": metric,
            "Mean_Bridge_Effect_Alone": mean_alone,
            "Mean_Bridge_Effect_With_Trust": mean_with_trust,
            "Cohen_d_Interaction": d,
            "Effect_Magnitude": d_magnitude(d),
            "W_statistic": w_stat,
            "p_value_raw": p_val,
            "Trust_KPI_Structural": "Yes" if note else "No",
            "Note": note,
        })

    results_df = pd.DataFrame(interaction_rows)
    if not results_df.empty:
        adjusted = bonferroni_correct(results_df["p_value_raw"].tolist())
        results_df["p_value_adjusted"] = adjusted
        results_df["Bonferroni_m"] = len(adjusted)
        results_df["Significant_0.05"] = [
            "Yes" if p < 0.05 else "No" for p in adjusted
        ]

    return results_df


def format_table(pairwise_df, interaction_df):
    print()
    print("=" * 110)
    print("  DYNAMIC-BABE — PAIRWISE (seed-paired Wilcoxon primary; MWU supplementary)")
    print("  Cohen's d = (Mean_Group1 - Mean_Group2) / pooled_SD")
    print("=" * 110)

    m_pair = int(pairwise_df["Bonferroni_m"].iloc[0]) if len(pairwise_df) else 0
    n_deg = int((pairwise_df["Degenerate"] == "Yes").sum()) if len(pairwise_df) else 0

    comparisons = pairwise_df["Comparison"].unique()
    for comp in comparisons:
        g1, g2 = comp.split(" vs ", 1)
        print(f"\n>>> Comparison: {_display(g1)} vs {_display(g2)}")
        print("-" * 110)
        subset = pairwise_df[pairwise_df["Comparison"] == comp]
        for _, row in subset.iterrows():
            if row["Effect_Size"] == "none":
                continue
            sig_mark = "  [** SIG **]" if row["Significant_0.05"] == "Yes" else ""
            print(
                f"  {row['Metric']:<22} | "
                f"Mean1={row['Mean1']:.4f} vs Mean2={row['Mean2']:.4f} | "
                f"d={row['Cohen_d']:+.4f} ({row['Effect_Size']:<10}) | "
                f"W={row['W_statistic']:.1f} adj_p={row['p_value_adjusted']:.6f}"
                f"{sig_mark}"
            )

    print("\n" + "=" * 110)
    print("  FILTER × TRUST INTERACTION (Wilcoxon on paired contrasts)")
    print("=" * 110)
    print(
        f"  {'Metric':<22} | {'Filter Alone':<14} | {'Filter w/Trust':<14} | "
        f"{'d_int':<8} | {'W':<8} | {'adj_p':<8} | Sig"
    )
    print("-" * 110)

    for _, row in interaction_df.iterrows():
        sig_mark = "Yes" if row["Significant_0.05"] == "Yes" else "No"
        flag = " [trust-KPI]" if row.get("Trust_KPI_Structural") == "Yes" else ""
        print(
            f"  {row['Metric']:<22} | "
            f"{row['Mean_Bridge_Effect_Alone']:+13.4f} | "
            f"{row['Mean_Bridge_Effect_With_Trust']:+13.4f} | "
            f"{row['Cohen_d_Interaction']:+7.4f} | "
            f"{row['W_statistic']:<8.1f} | "
            f"{row['p_value_adjusted']:.6f} | {sig_mark}{flag}"
        )
    print("=" * 110)
    print(
        f"  * Primary pairwise: seed-paired Wilcoxon; Bonferroni m={m_pair} "
        f"(excludes {n_deg} degenerate flat contrasts)."
    )
    print("  * MWU retained in CSV as supplementary unpaired sensitivity.")
    print("  * Interaction Bonferroni applied separately from pairwise family.")
    print(
        "  * Trust-KPI interactions with Delta_alone=0 are design artifacts "
        "(flagged); not comparable to retention moderation."
    )
    print("  * Trust_Segregation capped at 1e6 for analysis.")
    print("=" * 110)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading final-step data from 2×2 factorial batch runs...")
    data_dict = load_final_step_data()

    if not data_dict:
        print("  [ERROR] No data could be loaded. Please run batch_run.py first.")
        return

    print("Running pairwise comparisons (seed-paired Wilcoxon primary)...")
    pairwise_results = run_analysis(data_dict)

    print("Running Filter × Trust interaction analysis...")
    interaction_results = run_interaction_analysis(data_dict)

    format_table(pairwise_results, interaction_results)

    pairwise_results.to_csv(
        os.path.join(OUT_DIR, "statistical_analysis.csv"), index=False
    )
    interaction_results.to_csv(
        os.path.join(OUT_DIR, "interaction_analysis.csv"), index=False
    )

    print()
    print(
        f"  Pairwise results exported to: "
        f"{os.path.join(OUT_DIR, 'statistical_analysis.csv')}"
    )
    print(
        f"  Interaction results exported to: "
        f"{os.path.join(OUT_DIR, 'interaction_analysis.csv')}"
    )


if __name__ == "__main__":
    main()

"""
Diagnostic: count assimilation updates where the *raw* ratio
mu_raw = w_ij / (1+beta) exceeds 1 (before the [0,1] clamp).

Compares trust OFF vs trust ON (filter/bridge OFF), 3 seeds, full MAX_STEPS.
Also reports the post-clamp applied rate (should be 0% by construction).
"""
from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import config as cfg
from agent import SocialAgent
from model import SocialNetworkModel

SEEDS = [0, 1, 2]


def run_condition(enable_trust: bool, seeds=SEEDS):
    """Patch assimilate to count raw and applied mu>1, run full sims, restore."""
    total_assim = 0
    raw_over = 0
    applied_over = 0
    orig = SocialAgent.assimilate

    def counting_assimilate(self, other, w_ij):
        nonlocal total_assim, raw_over, applied_over
        mu_raw = w_ij / (1.0 + self.beta)
        mu_applied = min(1.0, max(0.0, mu_raw))
        total_assim += 1
        if mu_raw > 1.0 + 1e-12:
            raw_over += 1
        if mu_applied > 1.0 + 1e-12:
            applied_over += 1
        return orig(self, other, w_ij)

    SocialAgent.assimilate = counting_assimilate
    try:
        for seed in seeds:
            model = SocialNetworkModel(
                seed=seed,
                enable_bridge=False,
                enable_trust=enable_trust,
            )
            for _ in range(cfg.MAX_STEPS):
                model.step()
    finally:
        SocialAgent.assimilate = orig

    raw_pct = 100.0 * raw_over / total_assim if total_assim else 0.0
    applied_pct = 100.0 * applied_over / total_assim if total_assim else 0.0
    return total_assim, raw_over, raw_pct, applied_over, applied_pct


def main():
    print(f"MAX_STEPS={cfg.MAX_STEPS}, seeds={SEEDS}")
    print()

    lines = []
    for trust in (False, True):
        total, raw_n, raw_pct, app_n, app_pct = run_condition(enable_trust=trust)
        label = f"enable_trust={trust}, enable_bridge=False"
        block = (
            f"{label}:\n"
            f"  assimilation updates: {total}\n"
            f"  raw mu > 1:           {raw_n} ({raw_pct:.6f}%)\n"
            f"  applied mu > 1:       {app_n} ({app_pct:.6f}%)"
        )
        print(block)
        print()
        lines.append(
            f"trust_{'ON' if trust else 'OFF'} raw_mu>1: "
            f"{raw_pct:.6f}% ({raw_n}/{total}); "
            f"applied_mu>1: {app_pct:.6f}% ({app_n}/{total})"
        )

    out_dir = os.path.join(_ROOT, "output")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "mu_overshoot_diagnostic.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

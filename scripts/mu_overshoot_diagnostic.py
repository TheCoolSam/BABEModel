"""
Diagnostic: count assimilation updates where mu = w_ij / (1+beta) > 1.

Compares trust OFF vs trust ON (bridge OFF), 3 seeds, full MAX_STEPS.
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
    """Patch assimilate to count mu>1, run full sims, restore."""
    total_assim = 0
    mu_over = 0
    orig = SocialAgent.assimilate

    def counting_assimilate(self, other, w_ij):
        # Applied step size after clamp (must be <= 1 for DeGroot convexity).
        nonlocal total_assim, mu_over
        mu = min(1.0, max(0.0, w_ij / (1.0 + self.beta)))
        total_assim += 1
        if mu > 1.0 + 1e-12:
            mu_over += 1
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

    pct = 100.0 * mu_over / total_assim if total_assim else 0.0
    return total_assim, mu_over, pct


def main():
    print(f"MAX_STEPS={cfg.MAX_STEPS}, seeds={SEEDS}")
    print()

    off_total, off_over, off_pct = run_condition(enable_trust=False)
    print(
        f"enable_trust=False, enable_bridge=False:\n"
        f"  assimilation updates: {off_total}\n"
        f"  mu > 1:               {off_over}\n"
        f"  percentage:           {off_pct:.6f}%"
    )
    print()

    on_total, on_over, on_pct = run_condition(enable_trust=True)
    print(
        f"enable_trust=True, enable_bridge=False:\n"
        f"  assimilation updates: {on_total}\n"
        f"  mu > 1:               {on_over}\n"
        f"  percentage:           {on_pct:.6f}%"
    )

    out_dir = os.path.join(_ROOT, "output")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "mu_overshoot_diagnostic.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"trust_OFF mu>1: {off_pct:.6f}% ({off_over}/{off_total})\n")
        f.write(f"trust_ON  mu>1: {on_pct:.6f}% ({on_over}/{on_total})\n")
    print()
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

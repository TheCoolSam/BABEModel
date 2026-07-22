"""
Bit-check: two runs with the same seed must yield identical key KPIs.

  python scripts/check_reproducibility.py
"""

from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import config as cfg
from model import SocialNetworkModel

KEY_COLS = (
    "Polarization",
    "Polarization_All",
    "Churn_Rate",
    "Revenue",
    "Active_Users",
    "Avg_Frustration",
)


def _run(seed: int, enable_bridge: bool = False, enable_trust: bool = False):
    model = SocialNetworkModel(
        num_agents=cfg.NUM_AGENTS,
        ba_m=cfg.BA_EDGE_PARAM,
        enable_bridge=enable_bridge,
        enable_trust=enable_trust,
        seed=seed,
    )
    for _ in range(cfg.MAX_STEPS):
        if not model.running:
            break
        model.step()
    df = model.datacollector.get_model_vars_dataframe()
    return df[list(KEY_COLS)].copy()


def main():
    seed = cfg.RANDOM_SEED
    print(f"Running paired simulations with seed={seed} ...")
    a = _run(seed)
    b = _run(seed)
    if a.equals(b):
        print("PASS: identical Polarization / Polarization_All / Churn / Revenue / ...")
        return 0
    print("FAIL: trajectories differ under the same seed")
    for col in KEY_COLS:
        if not a[col].equals(b[col]):
            print(f"  mismatch in {col}: final {a[col].iloc[-1]} vs {b[col].iloc[-1]}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

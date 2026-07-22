"""Shared helpers for mechanism / baseline / topology experiment CLIs."""

from __future__ import annotations

import argparse
import os
import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

import config as cfg
from model import SocialNetworkModel

KPIS = [
    "Polarization",
    "Churn_Rate",
    "Revenue",
    "Avg_Frustration",
    "Active_Users",
    "Mean_Trust",
    "Ingroup_Trust",
    "Outgroup_Trust",
    "Trust_Segregation",
    "Opinion_Clustering",
]


def parse_common_args(description: str) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--iters", type=int, default=20, help="Replications per cell")
    p.add_argument("--workers", type=int, default=2, help="Pool workers")
    p.add_argument("--outdir", type=str, default=None, help="Output directory")
    p.add_argument("--checkpoint-every", type=int, default=10)
    p.add_argument("--quick", action="store_true", help="iters=3 smoke test")
    p.add_argument("--laptop", action="store_true", help="Cap workers at 2")
    return p.parse_args()


def resolve_iters_workers(args: argparse.Namespace) -> tuple[int, int]:
    iters = 3 if args.quick else args.iters
    workers = args.workers
    if args.laptop:
        workers = min(workers, 2)
    workers = max(1, workers)
    return iters, workers


def out_paths(args: argparse.Namespace, stem: str) -> tuple[str, str]:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    outdir = args.outdir or os.path.join(root, "output")
    ckdir = os.path.join(outdir, "checkpoints")
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(ckdir, exist_ok=True)
    return (
        os.path.join(outdir, f"{stem}.csv"),
        os.path.join(ckdir, f"{stem}_partial.csv"),
    )


def run_one(kwargs: Dict[str, Any], max_steps: Optional[int] = None) -> Dict[str, Any]:
    """Run one simulation; kwargs passed to SocialNetworkModel (+ meta labels)."""
    meta = {
        k: kwargs.pop(k)
        for k in list(kwargs.keys())
        if k.startswith("meta_")
    }
    seed = kwargs.get("seed", cfg.RANDOM_SEED)
    model = SocialNetworkModel(**kwargs)
    steps = cfg.MAX_STEPS if max_steps is None else max_steps
    for _ in range(steps):
        if not model.running:
            break
        model.step()
    df = model.datacollector.get_model_vars_dataframe()
    last = df.iloc[-1]
    row = {k: float(last[k]) if k in last.index else np.nan for k in KPIS}
    # Cap absurd segregation for CSV stability
    if np.isfinite(row["Trust_Segregation"]) and row["Trust_Segregation"] > 1e6:
        row["Trust_Segregation"] = 1e6
    row["Seed"] = int(seed)
    for k, v in meta.items():
        row[k.replace("meta_", "", 1)] = v
    return row


def append_checkpoint(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    header = not os.path.exists(path)
    df.to_csv(path, mode="a", header=header, index=False)


def finalize_csv(partial_path: str, final_path: str) -> pd.DataFrame:
    df = pd.read_csv(partial_path) if os.path.exists(partial_path) else pd.DataFrame()
    if len(df):
        df.to_csv(final_path, index=False)
    return df

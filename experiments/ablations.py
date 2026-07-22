"""
Mechanism ablations for the Polarization Paradox.

Conditions M0–M5 (see plan). Checkpointed CSV for Colab/SSH resilience.

  python -m experiments.ablations --laptop --iters=20
  python -m experiments.ablations --quick
"""

from __future__ import annotations

import multiprocessing as mp
import os
import sys
import time

# Allow `python experiments/ablations.py` and `python -m experiments.ablations`
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import config as cfg
from experiments._common import (
    append_checkpoint,
    finalize_csv,
    out_paths,
    parse_common_args,
    resolve_iters_workers,
    run_one,
)


def _worker(job: dict) -> dict:
    return run_one(dict(job))


def build_jobs(iters: int) -> list:
    base = cfg.RANDOM_SEED
    specs = [
        # M0: current Bridge Only
        dict(
            meta_Condition="M0_bridge_intercept",
            enable_bridge=True,
            enable_trust=False,
            bridge_mode="intercept",
        ),
        # M1: placebo heal without intercepting opinion
        dict(
            meta_Condition="M1_heal_only",
            enable_bridge=True,
            enable_trust=False,
            bridge_mode="heal_only",
        ),
        # M2: drop interaction
        dict(
            meta_Condition="M2_bridge_drop",
            enable_bridge=True,
            enable_trust=False,
            bridge_mode="drop",
        ),
        # M3: intercept but no healing bonus
        dict(
            meta_Condition="M3_no_heal_bonus",
            enable_bridge=True,
            enable_trust=False,
            bridge_mode="intercept",
            bridge_healing_bonus=0,
        ),
        # M4: Full Model without trust decay
        dict(
            meta_Condition="M4_full_no_decay",
            enable_bridge=True,
            enable_trust=True,
            bridge_mode="intercept",
            disable_trust_decay=True,
            trust_decay=0.0,
        ),
        # M5: Bridge Only, no brand-safety cliff
        dict(
            meta_Condition="M5_no_brand_safety",
            enable_bridge=True,
            enable_trust=False,
            bridge_mode="intercept",
            unsafe_ad_multiplier=1.0,
        ),
        # Reference: unmoderated baseline (for paradox contrasts)
        dict(
            meta_Condition="REF_baseline",
            enable_bridge=False,
            enable_trust=False,
            bridge_mode="intercept",
        ),
    ]
    jobs = []
    for i in range(iters):
        seed = base + i
        for spec in specs:
            job = dict(spec)
            job["seed"] = seed
            job["meta_Run"] = i
            jobs.append(job)
    return jobs


def main():
    args = parse_common_args("Mechanism ablations")
    iters, workers = resolve_iters_workers(args)
    final_path, partial_path = out_paths(args, "ablation_results")
    if os.path.exists(partial_path):
        os.remove(partial_path)

    jobs = build_jobs(iters)
    print(f"Ablations: {len(jobs)} jobs, workers={workers}", flush=True)
    t0 = time.perf_counter()
    buf = []
    done = 0
    with mp.Pool(processes=workers) as pool:
        for row in pool.imap_unordered(_worker, jobs, chunksize=1):
            buf.append(row)
            done += 1
            if len(buf) >= args.checkpoint_every:
                append_checkpoint(partial_path, buf)
                buf = []
                print(f"  checkpoint {done}/{len(jobs)}", flush=True)
    append_checkpoint(partial_path, buf)
    df = finalize_csv(partial_path, final_path)
    print(f"Wrote {final_path} ({len(df)} rows) in {time.perf_counter()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()

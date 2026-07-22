"""
Baseline model comparison: Dynamic-BABE vs HK vs degroot_bias.

  python -m experiments.baselines --laptop --iters=20
"""

from __future__ import annotations

import multiprocessing as mp
import os
import sys
import time

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
        dict(
            meta_Model="dynamic_babe",
            opinion_model="dynamic_babe",
            enable_bridge=False,
            enable_trust=False,
        ),
        dict(
            meta_Model="hk",
            opinion_model="hk",
            enable_bridge=False,
            enable_trust=False,
        ),
        dict(
            meta_Model="degroot_bias",
            opinion_model="degroot_bias",
            enable_bridge=False,
            enable_trust=False,
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
    args = parse_common_args("Baseline model comparison")
    iters, workers = resolve_iters_workers(args)
    final_path, partial_path = out_paths(args, "baseline_compare")
    if os.path.exists(partial_path):
        os.remove(partial_path)

    jobs = build_jobs(iters)
    print(f"Baselines: {len(jobs)} jobs, workers={workers}", flush=True)
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

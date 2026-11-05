"""Experiment driver: plan / run (sharded, resumable) / aggregate / plot.

    python -m pmsim.experiment plan      configs/recovery.yaml
    python -m pmsim.experiment run       configs/recovery.yaml --shard 3 --nshards 64 --workers 64
    python -m pmsim.experiment aggregate configs/recovery.yaml
    python -m pmsim.experiment plot      configs/recovery.yaml

A task is one planted instance (n, C, rep).  Seeds are a deterministic function
of (seed, n, C, rep), so any task can be recomputed bit-for-bit, shards can be
re-run after preemption (finished tasks are skipped), and the instance for
(n, C, rep) is the same for the MLE and every chain.  Each shard appends one
JSON line per finished task to  <out_dir>/raw/shard_XXXX.jsonl  (few files, so
the Torch inode quota is not an issue).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from fractions import Fraction
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import yaml

from .estimators import DEFAULTS, chain_lengths


# ----------------------------------------------------------------------------
# config / tasks
# ----------------------------------------------------------------------------
def load_config(path):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg.setdefault("seed", 20261105)
    cfg.setdefault("out_dir", f"results/{Path(path).stem}")
    cfg["out_dir"] = os.path.expandvars(cfg["out_dir"])
    cfg["sampler"] = {**DEFAULTS, **cfg.get("sampler", {})}
    cfg["_path"] = str(path)
    return cfg


def task_key(n, C, rep):
    return f"n{n}_C{C:.6g}_r{rep}"


def enumerate_tasks(cfg):
    tasks = []
    seen = set()
    for block in cfg["grids"]:
        for n in block["n"]:
            for C in block["C"]:
                C = float(Fraction(str(C)))      # allows "2/3" in YAML
                for rep in range(int(block["reps"])):
                    k = task_key(int(n), C, rep)
                    if k not in seen:
                        seen.add(k)
                        tasks.append((int(n), C, rep))
    return tasks


def task_cost(n, C, cfg):
    """Rough cost model (seconds on one Xeon core); only used for balancing
    shards and for `plan`."""
    s = cfg["sampler"]
    b, m = chain_lengths(n, s)
    chains = s["n_random"] + s["n_mle"]
    per_step = 2.0e-6 if s["kind"] == "walk" else 2.5e-7
    return chains * (b + m) * per_step + 4e-8 * n * n * (s["sinkhorn_iters"] + 10)


def task_mem_bytes(n, cfg):
    s = cfg["sampler"]
    G = (s["n_random"] > 0) + (s["n_mle"] > 0)
    # W f32 + LSA f64 copy (transient) + Q f64 + alias 2xf32 + counts + sparse
    return n * n * (4 + 8 + 8 + 8 + 4 * G + 4) + (64 << 20)


def shard_tasks(cfg, shard, nshards):
    """Longest-processing-time-first assignment (deterministic)."""
    tasks = enumerate_tasks(cfg)
    order = sorted(tasks, key=lambda t: (-task_cost(t[0], t[1], cfg), t))
    load = np.zeros(nshards)
    mine = []
    for t in order:
        s = int(np.argmin(load))
        load[s] += task_cost(t[0], t[1], cfg)
        if s == shard:
            mine.append(t)
    return mine, load


def seed_for(cfg, n, C, rep):
    return np.random.SeedSequence([int(cfg["seed"]), int(n),
                                   int(round(C * 1_000_000)), int(rep)])


# ----------------------------------------------------------------------------
# worker
# ----------------------------------------------------------------------------
def _worker_init():
    for v in ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "MKL_NUM_THREADS"):
        os.environ.setdefault(v, "1")


def _run_task(args):
    n, C, rep, cfg = args
    from .estimators import run_instance
    row = run_instance(n, C, seed_for(cfg, n, C, rep), cfg["sampler"])
    row.update(rep=rep, key=task_key(n, C, rep))
    return row


def _done_keys(raw_dir):
    keys = set()
    for f in Path(raw_dir).glob("shard_*.jsonl"):
        with open(f) as fh:
            for line in fh:
                try:
                    keys.add(json.loads(line)["key"])
                except (json.JSONDecodeError, KeyError):
                    pass          # a line cut by preemption; task is redone
    return keys


def _available_mem():
    for var, mult in (("SLURM_MEM_PER_NODE", 1 << 20),):
        if os.environ.get(var):
            return int(os.environ[var]) * mult
    if os.environ.get("SLURM_MEM_PER_CPU") and os.environ.get("SLURM_CPUS_PER_TASK"):
        return (int(os.environ["SLURM_MEM_PER_CPU"])
                * int(os.environ["SLURM_CPUS_PER_TASK"]) << 20)
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError):
        return 1 << 40


def run(cfg, shard, nshards, workers):
    out = Path(cfg["out_dir"])
    raw = out / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    mine, _ = shard_tasks(cfg, shard, nshards)
    done = _done_keys(raw)
    todo = [t for t in mine if task_key(*t) not in done]
    print(f"[shard {shard}/{nshards}] {len(mine)} tasks, {len(todo)} to do",
          flush=True)
    if not todo:
        return

    # Respect the memory budget for the largest instance in this shard.
    n_max = max(t[0] for t in todo)
    mem_cap = max(1, int(0.85 * _available_mem() // task_mem_bytes(n_max, cfg)))
    workers = max(1, min(workers, mem_cap, len(todo)))
    print(f"[shard {shard}] workers={workers} (memory cap {mem_cap})", flush=True)

    # Compile (or load from NUMBA_CACHE_DIR) every kernel once, before forking
    # workers, so they never race on the on-disk cache.
    from .estimators import run_instance
    run_instance(16, 1.0, np.random.SeedSequence(0),
                 {**cfg["sampler"], "burn_factor": 1, "sample_factor": 1})

    path = raw / f"shard_{shard:04d}.jsonl"
    t0 = time.time()
    ctx = get_context("spawn")
    with open(path, "a") as fh, ProcessPoolExecutor(
            max_workers=workers, mp_context=ctx, initializer=_worker_init) as ex:
        futs = [ex.submit(_run_task, (n, C, rep, cfg)) for (n, C, rep) in todo]
        for k, fut in enumerate(as_completed(futs), 1):
            row = fut.result()
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
            if k % max(1, len(todo) // 20) == 0 or k == len(todo):
                print(f"[shard {shard}] {k}/{len(todo)} done "
                      f"({time.time() - t0:.0f}s)", flush=True)


# ----------------------------------------------------------------------------
# plan / aggregate
# ----------------------------------------------------------------------------
def plan(cfg, nshards=None, cpus=64):
    tasks = enumerate_tasks(cfg)
    tot = sum(task_cost(n, C, cfg) for n, C, _ in tasks)
    ns = sorted({t[0] for t in tasks})
    print(f"config      : {cfg['_path']}")
    print(f"out_dir     : {cfg['out_dir']}")
    print(f"tasks       : {len(tasks)}  (n in {ns})")
    print(f"sampler     : {cfg['sampler']}")
    for n in ns:
        b, m = chain_lengths(n, cfg["sampler"])
        cnt = sum(1 for t in tasks if t[0] == n)
        print(f"  n={n:6d}: {cnt:5d} tasks, burn {b:.2e} + sample {m:.2e} "
              f"proposals/chain, ~{task_cost(n, 2.0, cfg):7.1f} core-s/task, "
              f"~{task_mem_bytes(n, cfg) / 2**30:5.2f} GiB/worker")
    print(f"total       : ~{tot / 3600:.1f} core-hours (rough)")
    if nshards is None:
        nshards = max(1, min(64, math.ceil(tot / (cpus * 3600 * 2))))
    _, load = shard_tasks(cfg, 0, nshards)
    print(f"suggested   : {nshards} array tasks x {cpus} cpus, "
          f"~{load.max() / cpus / 3600:.2f} h wall each (rough)")
    return nshards


def aggregate(cfg):
    import pandas as pd
    out = Path(cfg["out_dir"])
    rows = []
    for f in sorted((out / "raw").glob("shard_*.jsonl")):
        with open(f) as fh:
            for line in fh:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    if not rows:
        sys.exit(f"no results under {out / 'raw'}")
    df = pd.DataFrame(rows).drop_duplicates("key", keep="last")
    df["diff"] = df["ov_bayes"] - df["ov_mle"]
    df["q_gap"] = df["q_rand"] - df["q_mle"]
    df.sort_values(["n", "C", "rep"]).to_csv(out / "results.csv", index=False)

    cols = ["ov_bayes", "ov_mle", "diff", "q", "ov_sample", "q_rand", "q_mle",
            "q_gap", "bayes_agree", "bayes_eq_mle", "ez_rand", "ez_mle", "acc",
            "t_total"]
    cols = [c for c in cols if c in df]
    g = df.groupby(["n", "C"])[cols]
    summ = g.mean().add_suffix("_mean").join(g.sem().add_suffix("_se"))
    summ["reps"] = g.size()
    summ.reset_index().to_csv(out / "summary.csv", index=False)
    print(f"{len(df)} tasks -> {out / 'results.csv'}, {out / 'summary.csv'}")
    return df


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m pmsim.experiment")
    ap.add_argument("cmd", choices=["plan", "run", "aggregate", "plot"])
    ap.add_argument("config")
    ap.add_argument("--shard", type=int,
                    default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    ap.add_argument("--nshards", type=int,
                    default=int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1)))
    ap.add_argument("--workers", type=int,
                    default=int(os.environ.get("SLURM_CPUS_PER_TASK",
                                               os.cpu_count() or 1)))
    ap.add_argument("--cpus", type=int, default=64,
                    help="cpus per array task assumed by `plan`")
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    if a.cmd == "plan":
        n = plan(cfg, None, a.cpus)
        print(f"NSHARDS={n}")
    elif a.cmd == "run":
        run(cfg, a.shard, a.nshards, a.workers)
    elif a.cmd == "aggregate":
        aggregate(cfg)
    elif a.cmd == "plot":
        from .plots import make_all
        make_all(cfg)


if __name__ == "__main__":
    main()

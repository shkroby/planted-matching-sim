"""Jerrum-Sinclair-Vigoda (JSV) Metropolis chain on partial matchings.

Target (Section "Algorithmic results" of the write-up):

    P_{mu,lambda}(M | W)  propto  exp( mu <Pi_M, W> + |M| log lambda ),  M in M_{<=n},

with penalty lambda = n^c.  One step (Definition def:jsv): draw an edge e=(u,v)
of K_{n,n} uniformly and propose

    (down)   e in M                           ->  M \\ {e},       ratio 1/(lambda e^{mu W_e})
    (up)     u, v both uncovered              ->  M + {e},        ratio lambda e^{mu W_e}
    (slide)  exactly one of u, v covered, by e' -> M + {e} - {e'}, ratio e^{mu (W_e - W_e')}
    (none)   otherwise

accepted with probability min(1, ratio).  The lazy version (stay w.p. 1/2) has
the same stationary law; it is available (``lazy=True``) but off by default
because it only halves the speed.

Estimators (the write-up's M~(W)):  the ergodic average
    P~_e = (1/K) sum_s 1{e in M_{T+s}}
over *all* sampling steps is rounded to the max-weight perfect matching.
Because the restriction of P_{mu,lambda} to perfect matchings is exactly the
posterior P_mu (Lemma lem:simple), we also record the average over the steps
at which M_t is perfect; rounding that gives a second estimator, which is
consistent for M_OPT for any lambda (if the chain visits M_n at all).

Bookkeeping is O(1) per accepted move:
* rows keep the step and the "perfect-clock" value at which their current edge
  was entered; on leaving the edge the holding time and the number of perfect
  steps in between are added to two integer counters;
* |M|, |M cap M*| and the perfect clock are updated incrementally;
* ``(none)`` moves are decided from the O(n) mate arrays without touching W,
  so most steps never leave the cache.
"""

from __future__ import annotations

import math
import time

import numpy as np
from numba import njit
from scipy.sparse import csr_matrix

from .estimators import DEFAULTS as SWAP_DEFAULTS
from .estimators import bayes_round, mle, run_instance
from .model import child_seed, overlap, sample_instance
from .swapchain import _S32, _U32, _bounded, _next, _unif, make_rng_state

JSV_DEFAULTS = dict(
    c_values=[0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0],  # lambda = n^c
    burn_sweeps=200.0,     # burn-in proposals per chain / n^2
    sample_sweeps=200.0,   # sampling proposals per chain / n^2
    n_empty=2,             # chains started at the empty matching (as in the write-up)
    n_mle=2,               # chains started at the MLE (perfect) matching
    lazy=False,
    reference_swap=True,   # also run the swap-chain Bayes-OPT on the same instance
)


@njit(cache=True, inline="always")
def _remove(r, i, c, mates, colm, size, ov, sigma, start, pstart, pclock,
            counts, cperf, g, s, sampling, track):
    if sampling and track:
        counts[g, i, c] += s - start[r, i]
        cperf[g, i, c] += pclock[r] - pstart[r, i]
    mates[r, i] = -1
    colm[r, c] = -1
    size[r] -= 1
    if c == sigma[i]:
        ov[r] -= 1


@njit(cache=True, inline="always")
def _add(r, i, c, mates, colm, size, ov, sigma, start, pstart, pclock, s):
    mates[r, i] = c
    colm[r, c] = i
    size[r] += 1
    if c == sigma[i]:
        ov[r] += 1
    start[r, i] = s
    pstart[r, i] = pclock[r]


@njit(cache=True)
def _jsv_chains(W, mu, loglam, mates, sigma, n_burn, n_sample, lazy, group,
                counts, cperf, rng):
    """R JSV chains in lock-step on one instance.

    mates : (R, n) int64, row -> column or -1 (modified in place).
    counts, cperf : (G, n, n) integer arrays (or (G, 1, 1) to skip): holding
        times of each edge over all sampling steps / over perfect steps only.
    Returns per-chain arrays (size_sum, perfect_steps, ov_sum, ov_perf_sum,
    n_up, n_down, n_slide) over the sampling phase.
    """
    R, n = mates.shape
    nn = np.uint64(n)
    colm = np.full((R, n), -1, dtype=np.int64)
    size = np.zeros(R, dtype=np.int64)
    ov = np.zeros(R, dtype=np.int64)
    for r in range(R):
        for i in range(n):
            c = mates[r, i]
            if c >= 0:
                colm[r, c] = i
                size[r] += 1
                if c == sigma[i]:
                    ov[r] += 1
    track = counts.shape[1] == n
    start = np.zeros((R, n), dtype=np.int64)
    pstart = np.zeros((R, n), dtype=np.int64)
    pclock = np.zeros(R, dtype=np.int64)
    size_sum = np.zeros(R, dtype=np.int64)
    ov_sum = np.zeros(R, dtype=np.int64)
    ov_perf = np.zeros(R, dtype=np.int64)
    n_up = np.zeros(R, dtype=np.int64)
    n_down = np.zeros(R, dtype=np.int64)
    n_slide = np.zeros(R, dtype=np.int64)

    total = n_burn + n_sample
    for t in range(total):
        sampling = t >= n_burn
        s = t - n_burn
        if t == n_burn:                       # edges held at the start of sampling
            for r in range(R):
                pclock[r] = 0
                for i in range(n):
                    start[r, i] = 0
                    pstart[r, i] = 0
        for r in range(R):
            g = group[r]
            x = _next(rng, r)
            u = _bounded(x >> _S32, nn, rng, r)
            v = _bounded(x & _U32, nn, rng, r)
            if lazy and _unif(rng, r) < 0.5:
                pass
            else:
                mu_u = mates[r, u]
                mv = colm[r, v]
                if mu_u == v:                                     # (down)
                    logr = -(mu * W[u, v] + loglam)
                    if logr >= 0.0 or _unif(rng, r) < math.exp(logr):
                        _remove(r, u, v, mates, colm, size, ov, sigma, start,
                                pstart, pclock, counts, cperf, g, s, sampling,
                                track)
                        if sampling:
                            n_down[r] += 1
                elif mu_u < 0 and mv < 0:                         # (up)
                    logr = mu * W[u, v] + loglam
                    if logr >= 0.0 or _unif(rng, r) < math.exp(logr):
                        _add(r, u, v, mates, colm, size, ov, sigma, start,
                             pstart, pclock, s)
                        if sampling:
                            n_up[r] += 1
                elif mu_u < 0:                                    # (slide) v taken by mv
                    logr = mu * (W[u, v] - W[mv, v])
                    if logr >= 0.0 or _unif(rng, r) < math.exp(logr):
                        _remove(r, mv, v, mates, colm, size, ov, sigma, start,
                                pstart, pclock, counts, cperf, g, s, sampling,
                                track)
                        _add(r, u, v, mates, colm, size, ov, sigma, start,
                             pstart, pclock, s)
                        if sampling:
                            n_slide[r] += 1
                elif mv < 0:                                      # (slide) u has mu_u
                    logr = mu * (W[u, v] - W[u, mu_u])
                    if logr >= 0.0 or _unif(rng, r) < math.exp(logr):
                        _remove(r, u, mu_u, mates, colm, size, ov, sigma, start,
                                pstart, pclock, counts, cperf, g, s, sampling,
                                track)
                        _add(r, u, v, mates, colm, size, ov, sigma, start,
                             pstart, pclock, s)
                        if sampling:
                            n_slide[r] += 1
                # else: (none) move
            if sampling:
                size_sum[r] += size[r]
                ov_sum[r] += ov[r]
                if size[r] == n:
                    pclock[r] += 1
                    ov_perf[r] += ov[r]

    if track and n_sample > 0:
        for r in range(R):
            g = group[r]
            for i in range(n):
                c = mates[r, i]
                if c >= 0:
                    counts[g, i, c] += n_sample - start[r, i]
                    cperf[g, i, c] += pclock[r] - pstart[r, i]
    return size_sum, pclock, ov_sum, ov_perf, n_up, n_down, n_slide


def run_jsv(W, mu, loglam, init_mates, sigma, n_burn, n_sample, counts=None,
            cperf=None, group=None, lazy=False, seed=0):
    """Run R = len(init_mates) JSV chains (entries -1 = uncovered row).

    Returns (final mates (R, n), list of per-chain stats dicts)."""
    mates = np.ascontiguousarray(np.array(init_mates, dtype=np.int64))
    if mates.ndim == 1:
        mates = mates[None, :]
    R, n = mates.shape
    sigma = np.ascontiguousarray(sigma, dtype=np.int64)
    if counts is None:
        counts = np.zeros((1, 1, 1), dtype=np.int64)
        cperf = np.zeros((1, 1, 1), dtype=np.int64)
    group = np.zeros(R, np.int64) if group is None else np.asarray(group, np.int64)
    if not isinstance(seed, np.random.SeedSequence):
        seed = np.random.SeedSequence(int(seed))
    out = _jsv_chains(W, float(mu), float(loglam), mates, sigma, int(n_burn),
                      int(n_sample), bool(lazy), group, counts, cperf,
                      make_rng_state(seed, R))
    size_sum, pclock, ov_sum, ov_perf, n_up, n_down, n_slide = out
    K = max(int(n_sample), 1)
    stats = []
    for r in range(R):
        P = int(pclock[r])
        stats.append(dict(
            mean_size=size_sum[r] / K / n,
            frac_perfect=P / K,
            q_all=ov_sum[r] / K / n,                     # <|M cap M*|>/n, all steps
            q_perf=(ov_perf[r] / P / n) if P else np.nan,  # same, perfect steps only
            perfect_steps=P,
            rate_up=n_up[r] / K, rate_down=n_down[r] / K, rate_slide=n_slide[r] / K,
            final_size=int(np.sum(mates[r] >= 0)),
        ))
    return mates, stats


# ----------------------------------------------------------------------------
# One instance: MLE, swap-chain Bayes-OPT (reference) and JSV for every lambda
# ----------------------------------------------------------------------------
def run_jsv_instance(n: int, C: float, seed_seq: np.random.SeedSequence,
                     cfg: dict | None = None, swap_cfg: dict | None = None):
    """Returns one row per penalty c (lambda = n^c), all on the same instance
    (same W as ``estimators.run_instance`` with the same seed)."""
    cfg = {**JSV_DEFAULTS, **(cfg or {})}
    swap_cfg = {**SWAP_DEFAULTS, **(swap_cfg or {})}
    t0 = time.perf_counter()

    ref = {}
    if cfg["reference_swap"]:
        r = run_instance(n, C, seed_seq, swap_cfg)
        ref = dict(ov_swap=r["ov_bayes"], q_swap=r["q"],
                   q_swap_gap=r["q_rand"] - r["q_mle"], t_swap=r["t_total"])

    W, sigma, mu = sample_instance(n, C, np.random.default_rng(child_seed(seed_seq, 0)))
    m_mle = mle(W)
    ov_mle = overlap(m_mle, sigma)

    ne, nm = int(cfg["n_empty"]), int(cfg["n_mle"])
    inits = [np.full(n, -1)] * ne + [m_mle] * nm
    G = (ne > 0) + (nm > 0)
    group = [0] * ne + [1 if G == 2 else 0] * nm
    n2 = n * n
    n_burn = int(math.ceil(cfg["burn_sweeps"] * n2))
    n_samp = int(math.ceil(cfg["sample_sweeps"] * n2))
    big = max(ne, nm) * n_samp >= 2**31 - 1
    dt = np.int64 if big else np.int32

    rows = []
    for k, c in enumerate(cfg["c_values"]):
        t1 = time.perf_counter()
        loglam = float(c) * math.log(n)
        counts = np.zeros((G, n, n), dtype=dt)
        cperf = np.zeros((G, n, n), dtype=dt)
        _, st = run_jsv(W, mu, loglam, inits, sigma, n_burn, n_samp,
                        counts=counts, cperf=cperf, group=group,
                        lazy=cfg["lazy"], seed=child_seed(seed_seq, 100 + k))
        S = [csr_matrix(counts[g]) for g in range(G)]
        Sp = [csr_matrix(cperf[g]) for g in range(G)]
        del counts, cperf
        m_all = bayes_round(sum(S[1:], S[0]), fallback=m_mle)
        m_grp = [bayes_round(Sg, fallback=m_mle) for Sg in S]
        Sp_tot = sum(Sp[1:], Sp[0])
        m_perf = bayes_round(Sp_tot, fallback=m_mle) if Sp_tot.nnz else None

        def gm(key, idx):
            v = [st[i][key] for i in idx]
            return float(np.nanmean(v)) if len(v) and not np.all(np.isnan(v)) else np.nan

        ie, im = list(range(ne)), list(range(ne, ne + nm))
        rows.append(dict(
            n=n, C=C, mu=mu, c=float(c), loglam=loglam,
            ov_mle=ov_mle, **ref,
            ov_jsv=overlap(m_all, sigma),                     # write-up's M~(W)
            ov_jsv_perf=overlap(m_perf, sigma) if m_perf is not None else np.nan,
            ov_jsv_empty=overlap(m_grp[0], sigma) if ne else np.nan,
            ov_jsv_mlestart=overlap(m_grp[-1], sigma) if nm else np.nan,
            jsv_agree=overlap(m_grp[0], m_grp[-1]) if G == 2 else 1.0,
            q_jsv=gm("q_all", ie + im), q_jsv_perf=gm("q_perf", ie + im),
            q_jsv_empty=gm("q_all", ie), q_jsv_mle=gm("q_all", im),
            frac_perfect=gm("frac_perfect", ie + im),
            frac_perfect_empty=gm("frac_perfect", ie),
            mean_size=gm("mean_size", ie + im),
            rate_up=gm("rate_up", ie + im), rate_down=gm("rate_down", ie + im),
            rate_slide=gm("rate_slide", ie + im),
            n_burn=n_burn, n_sample=n_samp,
            t_jsv=time.perf_counter() - t1,
        ))
    for row in rows:
        row["t_total"] = time.perf_counter() - t0
    return rows

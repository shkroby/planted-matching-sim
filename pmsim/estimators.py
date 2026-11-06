"""MLE and (MCMC-approximated) Bayes-optimal estimators for one instance.

Recovery is measured by the overlap  ov(M) = |M cap M*| / n = 1 - |M triangle M*|/(2n),
so the normalised error of the write-up is r_n = 1 - E[ov].
"""

from __future__ import annotations

import math
import time

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching

from .model import child_seed, overlap, sample_instance
from .swapchain import Proposal, run_chains

DEFAULTS = dict(
    kind="walk",            # "walk" (ejection chains) | "rotation" | "uniform"
    burn_factor=100.0,      # burn-in proposals per chain, in units of n ln n
    sample_factor=100.0,    # sampling proposals per chain, in units of n ln n
    n_random=2,             # chains started at a uniformly random matching
    n_mle=2,                # chains started at the MLE
    anneal=0.5,             # fraction of burn-in used to ramp C (random starts)
    eps=0.05,               # uniform-column mixing in the informed proposal
    sinkhorn_iters=30,      # column prices for the proposal (exactness-neutral)
    max_len=64,             # longest ejection chain ("walk")
    kmax=32,                # capped rejection for "column != current"
    cycle_probs=[0.4, 0.25, 0.2, 0.15],   # P(L = 2..5), "rotation" only
    n_trace=100,
)


# ----------------------------------------------------------------------------
# MLE: argmax_M <Pi_M, W>  (exact linear assignment)
# ----------------------------------------------------------------------------
def mle(W: np.ndarray) -> np.ndarray:
    _, col = linear_sum_assignment(W, maximize=True)
    return col.astype(np.int64)


# ----------------------------------------------------------------------------
# Bayes-optimal: argmax_M sum_{e in M} p_e(W)
# ----------------------------------------------------------------------------
def bayes_round(S, fallback=None) -> np.ndarray:
    """Max-weight perfect matching on the (sparse) estimated marginal matrix.

    The support of the visit counts contains every matching the chains visited,
    so a perfect matching exists inside it; the assignment problem is solved on
    that sparse support (LAPJVsp), much faster than a dense solve.

    ``fallback`` (a permutation) adds its edges with a negligible weight; it is
    needed when the counts come from *partial* matchings (JSV chain), whose
    support need not contain a perfect matching.  Rows that the chain never
    covered are then completed by the fallback edges.
    """
    if not isinstance(S, csr_matrix):
        S = csr_matrix(np.asarray(S, dtype=np.float64))
    S = S.astype(np.float64)
    if fallback is not None:
        n = S.shape[0]
        scale = max(float(S.max()) if S.nnz else 1.0, 1.0)
        S = S + csr_matrix((np.full(n, 1e-9 * scale), (np.arange(n), fallback)),
                           shape=S.shape)
    S.eliminate_zeros()
    _, col = min_weight_full_bipartite_matching(S, maximize=True)
    return col.astype(np.int64)


def chain_lengths(n: int, cfg: dict) -> tuple[int, int]:
    nlogn = n * math.log(n)
    return (int(math.ceil(cfg["burn_factor"] * nlogn)),
            int(math.ceil(cfg["sample_factor"] * nlogn)))


def run_instance(n: int, C: float, seed_seq: np.random.SeedSequence,
                 cfg: dict | None = None, keep_traces: bool = False) -> dict:
    """Simulate one planted instance and evaluate MLE vs Bayes-OPT.

    Two groups of chains are run in lock-step on the same instance: chains from
    a uniformly random matching (with an annealed burn-in) and chains from the
    MLE.  They approach the posterior from opposite sides (from below / above
    in overlap), so their agreement is the convergence diagnostic.  Marginals
    are pooled over all chains for the reported Bayes estimator.
    """
    cfg = {**DEFAULTS, **(cfg or {})}
    t0 = time.perf_counter()
    s_inst, s_chain, s_perm = (child_seed(seed_seq, i) for i in range(3))
    W, sigma, mu = sample_instance(n, C, np.random.default_rng(s_inst))
    t_gen = time.perf_counter() - t0

    t1 = time.perf_counter()
    m_mle = mle(W)
    t_mle = time.perf_counter() - t1

    t2 = time.perf_counter()
    prop = Proposal(W, mu, cfg["kind"], eps=cfg["eps"],
                    cycle_probs=tuple(cfg["cycle_probs"]),
                    sinkhorn_iters=int(cfg["sinkhorn_iters"]),
                    max_len=int(cfg["max_len"]), kmax=int(cfg["kmax"]))
    if cfg["kind"] != "uniform":
        del W                               # chains only need Q and the alias tables
    t_tab = time.perf_counter() - t2

    nr, nm = int(cfg["n_random"]), int(cfg["n_mle"])
    prng = np.random.default_rng(s_perm)
    inits = [prng.permutation(n) for _ in range(nr)] + [m_mle] * nm
    group = [0] * nr + [1] * nm
    anneal = [cfg["anneal"]] * nr + [0.0] * nm
    n_burn, n_samp = chain_lengths(n, cfg)
    G = (nr > 0) + (nm > 0)
    group = [g if G == 2 else 0 for g in group]
    big = max(nr, nm) * n_samp >= 2**31 - 1
    counts = np.zeros((G, n, n), dtype=np.int64 if big else np.int32)

    t3 = time.perf_counter()
    perms, stats = run_chains(prop.W, mu, inits, sigma, n_burn, n_samp,
                              counts=counts, group=group, anneal=anneal,
                              n_trace=cfg["n_trace"], seed=s_chain,
                              proposal=prop)
    t_mcmc = time.perf_counter() - t3
    del prop

    t4 = time.perf_counter()
    S = [csr_matrix(counts[g]) for g in range(G)]
    del counts
    m_groups = [bayes_round(Sg) for Sg in S]
    m_bayes = bayes_round(sum(S[1:], S[0]))
    t_round = time.perf_counter() - t4

    def gmean(key, idx):
        return float(np.mean([stats[r][key] for r in idx])) if idx else np.nan

    ir, im = list(range(nr)), list(range(nr, nr + nm))
    row = dict(
        n=n, C=C, mu=mu, n_burn=n_burn, n_sample=n_samp, chains=nr + nm,
        # --- recovery (overlap with the planted matching) ---
        ov_mle=overlap(m_mle, sigma),
        ov_bayes=overlap(m_bayes, sigma),
        ov_sample=gmean("final_overlap", ir + im),   # one posterior draw
        q=gmean("q_sample", ir + im),                # E<|M cap M*|>/n = 1 - mmse_n
        # --- per-start-group diagnostics ---
        q_rand=gmean("q_sample", ir), q_mle=gmean("q_sample", im),
        ov_bayes_rand=overlap(m_groups[0], sigma) if nr else np.nan,
        ov_bayes_mlestart=overlap(m_groups[-1], sigma) if nm else np.nan,
        bayes_agree=overlap(m_groups[0], m_groups[-1]) if G == 2 else 1.0,
        bayes_eq_mle=overlap(m_bayes, m_mle),
        ez_rand=gmean("energy_z", ir), ez_mle=gmean("energy_z", im),
        acc=gmean("acc_rate", ir + im),
        move_len=gmean("mean_move_len", ir + im) if cfg["kind"] == "walk" else 2.0,
        # --- timings (s) ---
        t_gen=t_gen, t_mle=t_mle, t_tables=t_tab, t_mcmc=t_mcmc,
        t_round=t_round, t_total=time.perf_counter() - t0,
    )
    if keep_traces:
        row["_traces"] = [(s["trace_ov"], s["trace_energy"]) for s in stats]
    return row

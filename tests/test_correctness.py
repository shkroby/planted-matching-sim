"""Exactness checks against brute-force enumeration on tiny instances.

Run with:  python -m pytest -q tests
"""

import itertools
import math

import numpy as np
from scipy.special import logsumexp

from pmsim.estimators import bayes_round, mle
from pmsim.model import sample_instance
from pmsim.swapchain import Proposal, run_chains, swap_chain


def brute_force_marginals(W, mu):
    n = W.shape[0]
    perms = np.array(list(itertools.permutations(range(n))))
    logw = mu * W[np.arange(n)[None, :], perms].sum(1)
    p = np.exp(logw - logsumexp(logw))
    P = np.zeros((n, n))
    for w, pi in zip(p, perms):
        P[np.arange(n), pi] += w
    best = perms[np.argmax(W[np.arange(n)[None, :], perms].sum(1))]
    return P, best


import pytest


@pytest.mark.parametrize("kind,cycles", [("uniform", (1.0,)),
                                          ("walk", (1.0,)),
                                          ("informed", (1.0,)),
                                          ("informed", (0.4, 0.25, 0.2, 0.15)),
                                          ("informed", (0.0, 0.0, 1.0))])  # 4-cycles only (odd => ergodic)
def test_marginals_match_enumeration(kind, cycles):
    rng = np.random.default_rng(0)
    # without 2-swaps the chain is ergodic but practically stuck at large C
    Cs = (0.5, 2.0, 4.0) if cycles[0] > 0 else (0.5, 2.0)
    for C in Cs:
        n = 7
        W, sigma, mu = sample_instance(n, C, rng, dtype=np.float64)
        P_exact, best = brute_force_marginals(W, mu)
        counts = np.zeros((1, n, n), dtype=np.int64)
        n_samp, R = 1_000_000, 4
        run_chains(W, mu, [rng.permutation(n) for _ in range(R)], sigma,
                   10_000, n_samp, counts=counts, seed=1,
                   proposal=Proposal(W, mu, kind, eps=0.05, cycle_probs=cycles))
        P_hat = counts[0] / (R * n_samp)
        assert np.allclose(P_hat.sum(0), 1) and np.allclose(P_hat.sum(1), 1)
        assert np.max(np.abs(P_hat - P_exact)) < 0.01, (C, np.max(np.abs(P_hat - P_exact)))
        assert np.array_equal(mle(W), best)


def test_planted_overlap_time_average():
    """q_sample must equal the time-averaged marginal mass on the planted edges."""
    rng = np.random.default_rng(1)
    n = 50
    W, sigma, mu = sample_instance(n, 1.5, rng, dtype=np.float64)
    counts = np.zeros((n, n), dtype=np.int64)
    _, st = swap_chain(W, mu, rng.permutation(n), sigma, 1000, 200_000,
                       counts=counts, seed=3)
    q_from_counts = counts[np.arange(n), sigma].sum() / (200_000 * n)
    assert math.isclose(st["q_sample"], q_from_counts, rel_tol=1e-12)


def test_bayes_round_is_perfect_matching():
    rng = np.random.default_rng(2)
    n = 40
    W, sigma, mu = sample_instance(n, 3.0, rng)
    counts = np.zeros((n, n), dtype=np.int32)
    swap_chain(W, mu, rng.permutation(n), sigma, 5000, 50_000, counts=counts)
    m = bayes_round(counts)
    assert sorted(m.tolist()) == list(range(n))


@pytest.mark.parametrize("kind", ["uniform", "rotation", "walk"])
def test_energy_bookkeeping(kind):
    rng = np.random.default_rng(4)
    n = 64
    W, sigma, mu = sample_instance(n, 2.0, rng, dtype=np.float64)
    p0 = rng.permutation(n)
    perm, st = swap_chain(W, mu, p0, sigma, 0, 300_000, n_trace=1,
                          proposal=Proposal(W, mu, kind))
    assert math.isclose(st["final_energy"], W[np.arange(n), perm].sum() / n,
                        rel_tol=1e-9, abs_tol=1e-9)
    assert st["final_overlap"] == np.mean(perm == sigma)


def test_multichain_matches_enumeration():
    """Lock-step chains with separate count groups and annealed burn-in."""
    from pmsim.swapchain import run_chains
    rng = np.random.default_rng(5)
    n = 6
    W, sigma, mu = sample_instance(n, 2.5, rng, dtype=np.float64)
    P_exact, _ = brute_force_marginals(W, mu)
    counts = np.zeros((2, n, n), dtype=np.int64)
    inits = [rng.permutation(n) for _ in range(4)]
    n_samp = 1_000_000
    run_chains(W, mu, inits, sigma, 20_000, n_samp, counts=counts,
               group=[0, 0, 1, 1], anneal=[0.5, 0.5, 0.0, 0.0],
               proposal=Proposal(W, mu, "rotation"))
    for g in range(2):
        assert np.max(np.abs(counts[g] / (2 * n_samp) - P_exact)) < 0.01


@pytest.mark.parametrize("eps,kmax", [(0.0, 1000), (0.05, 32), (0.2, 2)])
def test_walk_variants_match_enumeration(eps, kmax):
    """Ejection-chain kernel: rejection-free limit, capped rejection, annealing."""
    from pmsim.swapchain import run_chains
    rng = np.random.default_rng(6)
    for C in (1.0, 3.0):
        n = 7
        W, sigma, mu = sample_instance(n, C, rng, dtype=np.float64)
        P_exact, _ = brute_force_marginals(W, mu)
        counts = np.zeros((1, n, n), dtype=np.int64)
        R, n_samp = 4, 500_000
        run_chains(W, mu, [rng.permutation(n) for _ in range(R)], sigma, 20_000,
                   n_samp, counts=counts, anneal=[0.5, 0, 0.5, 0], seed=7,
                   proposal=Proposal(W, mu, "walk", eps=eps, kmax=kmax,
                                     max_len=5))
        assert np.max(np.abs(counts[0] / (R * n_samp) - P_exact)) < 0.01

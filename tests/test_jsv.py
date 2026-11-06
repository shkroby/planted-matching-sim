"""JSV chain: exactness against enumeration of all partial matchings."""

import itertools
import math

import numpy as np
import pytest
from scipy.special import logsumexp

from pmsim.jsv import run_jsv
from pmsim.model import sample_instance


def partial_matchings(n):
    """All partial matchings of K_{n,n} as row -> column arrays (-1 = free)."""
    out = []
    for k in range(n + 1):
        for rows in itertools.combinations(range(n), k):
            for cols in itertools.permutations(range(n), k):
                m = np.full(n, -1)
                m[list(rows)] = cols
                out.append(m)
    return np.array(out)


def exact(W, mu, loglam):
    n = W.shape[0]
    Ms = partial_matchings(n)
    covered = Ms >= 0
    logw = np.array([mu * W[np.arange(n)[c], m[c]].sum() + loglam * c.sum()
                     for m, c in zip(Ms, covered)])
    p = np.exp(logw - logsumexp(logw))
    P = np.zeros((n, n))
    Pperf = np.zeros((n, n))
    perfect = covered.all(1)
    for w, m, c, f in zip(p, Ms, covered, perfect):
        P[np.arange(n)[c], m[c]] += w
        if f:
            Pperf[np.arange(n), m] += w
    return P, Pperf / p[perfect].sum(), p[perfect].sum()


@pytest.mark.parametrize("C,c,lazy", [(1.0, 0.0, False), (2.0, 1.0, False),
                                      (3.0, 0.5, True), (0.5, 2.0, False)])
def test_jsv_marginals(C, c, lazy):
    rng = np.random.default_rng(11)
    n = 4
    assert len(partial_matchings(n)) == 209
    W, sigma, mu = sample_instance(n, C, rng, dtype=np.float64)
    loglam = c * math.log(n)
    P, Pperf, pperf = exact(W, mu, loglam)
    G, R, K = 1, 4, 1_000_000
    counts = np.zeros((G, n, n), np.int64)
    cperf = np.zeros((G, n, n), np.int64)
    inits = [np.full(n, -1), np.full(n, -1), rng.permutation(n), rng.permutation(n)]
    _, st = run_jsv(W, mu, loglam, inits, sigma, 50_000, K, counts=counts,
                    cperf=cperf, lazy=lazy, seed=3)
    assert np.max(np.abs(counts[0] / (R * K) - P)) < 0.01
    frac = np.mean([s["frac_perfect"] for s in st])
    assert abs(frac - pperf) < 0.01
    # restriction to perfect matchings is the posterior P_mu (Lemma lem:simple)
    assert np.max(np.abs(cperf[0] / cperf[0].sum(1, keepdims=True) - Pperf)) < 0.02
    # time-averaged planted overlap equals the marginal mass on sigma
    q = np.mean([s["q_all"] for s in st])
    assert abs(q - P[np.arange(n), sigma].mean()) < 0.01


def test_jsv_bookkeeping_consistency():
    rng = np.random.default_rng(12)
    n = 30
    W, sigma, mu = sample_instance(n, 2.0, rng)
    counts = np.zeros((1, n, n), np.int64)
    cperf = np.zeros((1, n, n), np.int64)
    K = 200_000
    mates, st = run_jsv(W, mu, 1.5 * math.log(n), [np.full(n, -1)], sigma,
                        10_000, K, counts=counts, cperf=cperf, seed=5)
    s = st[0]
    # sum of holding times = time-integrated matching size
    assert counts.sum() == round(s["mean_size"] * n * K)
    # every row is covered at every perfect step
    assert np.all(cperf[0].sum(1) == s["perfect_steps"])
    assert s["final_size"] == int((mates[0] >= 0).sum())

"""Asymptotic predictions from the write-up, used as reference curves."""

from __future__ import annotations

import math

import numpy as np

_trapz = getattr(np, "trapezoid", None) or np.trapz


def bayes_aon_overlap(C):
    """All-or-nothing limit (Thm. aon): E|M_OPT cap M*|/n -> 1{C > 2}."""
    return np.where(np.asarray(C) > 2, 1.0, 0.0)


def mle_error_exponent(C):
    """r_n(C; M_MLE) = n^{-delta(C) + o(1)}  (Dai-Cullina-Kiyavash).

    delta(C) = 1/2 - sqrt(C(4-C))/4 for 2 < C < 4 (matching converse for any
    estimator), and the bound r_n <~ n^{1 - C/2} for C > 4.
    """
    C = np.asarray(C, dtype=float)
    out = np.full_like(C, np.nan)
    mid = (C > 2) & (C < 4)
    out[mid] = 0.5 - 0.25 * np.sqrt(C[mid] * (4 - C[mid]))
    hi = C >= 4
    out[hi] = C[hi] / 2 - 1
    return out


def nu(C):
    return 2 - (1 + C / 2) ** 2 / (2 * C)


def qn_leading(n, C):
    """Leading term of q_n = 1 - mmse_n (Prop. mmse-asymptotics), 0 < C < 2."""
    n = np.asarray(n, dtype=float)
    if abs(C - 2 / 3) < 1e-9:
        return 0.5 * n ** (-1 / 3)
    if C < 2 / 3:
        return n ** (C - 1)
    a = math.pi / math.sin(math.pi * (1 / C - 0.5))
    return a / math.sqrt(2 * math.pi * C) * n ** (nu(C) - 1) / np.sqrt(np.log(n))


def qn_scalar(n, C, grid=20001):
    """Exact finite-n scalar reference n*rho(t), t = C log n.

    rho(t) = E[pbar(W)^2] for the Bernoulli(1/n) surrogate W = sqrt(t) B + Z,
    pbar(w) = P(B = 1 | W = w).  Since E[pbar^2] = E[pbar B],
        n rho = int phi(w - sqrt t) pbar(w) dw.
    """
    out = []
    for nn in np.atleast_1d(n):
        s = math.sqrt(C * math.log(nn))
        w = np.linspace(s - 12, s + 12, grid)
        pbar = 1.0 / (1.0 + (nn - 1) * np.exp(-s * w + s * s / 2))
        dens = np.exp(-0.5 * (w - s) ** 2) / math.sqrt(2 * math.pi)
        out.append(_trapz(dens * pbar, w))
    return np.array(out)

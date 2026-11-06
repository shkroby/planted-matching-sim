"""Figures for the JSV addendum (penalty lambda = n^c on partial matchings).

    jsv_recovery_vs_c   overlap of the JSV estimator M~(W) (all steps, as in the
                        write-up) and of its perfect-step variant vs c, with the
                        swap-chain Bayes-OPT and the MLE on the same instances
    jsv_gap_vs_c        ov(M~_JSV) - ov(M_OPT^swap) vs c, one line per n
    jsv_occupancy       fraction of steps on perfect matchings and E|M|/n vs c
    jsv_diagnostics     empty-start vs MLE-start JSV chains (convergence)
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import style  # noqa: E402
from .plots import Z, _fmt_C, _legend_below, _save  # noqa: E402
from .style import BLUE, CYCLE, GREEN, GREY, RUST, SEQ, marker  # noqa: E402


def _s(df, col, by=("c",)):
    g = df.groupby(list(by))[col]
    return pd.DataFrame({"m": g.mean(), "se": g.sem().fillna(0.0)}).reset_index()


def _c_label():
    return r"penalty exponent $c$  ($\lambda = n^{c}$)"


def _lemma_c(C):
    """Lemma peeling-1: TV(P_{mu,lambda}|_{M_n}, P_{mu,lambda}) = o(n^{-c0}) once
    c > 1 + sqrt(6C) + c0; we mark the c0 -> 0 end of that range."""
    return 1 + math.sqrt(6 * C)


def _hline(ax, d, col, color, ls, label):
    """Instance-average of a c-independent estimator (paired differences are
    in jsv_gap_vs_c, so no between-instance band is drawn here)."""
    ax.axhline(d[col].mean(), color=color, ls=ls, lw=1.6, label=label, zorder=2)


def _panels(k):
    fig, axes = plt.subplots(1, k, figsize=(5.0 * k, 4.3), squeeze=False)
    return fig, axes[0]


def recovery_vs_c(df, out, n, Cs):
    fig, axes = _panels(len(Cs))
    for ax, C in zip(axes, Cs):
        d = df[(df.n == n) & np.isclose(df.C, C)]
        inst = d.drop_duplicates("key")
        if "ov_swap" in d:
            _hline(ax, inst, "ov_swap", GREY, ":", "Bayes-OPT (swap chain)")
        _hline(ax, inst, "ov_mle", RUST, "--", "MLE")
        s = _s(d, "ov_jsv")
        ax.errorbar(s.c, s.m, yerr=Z * s.se, label=r"JSV $\widetilde M(W)$ (all steps)",
                    zorder=4, **marker(BLUE))
        s = _s(d.dropna(subset=["ov_jsv_perf"]), "ov_jsv_perf")
        ax.errorbar(s.c, s.m, yerr=Z * s.se, label="JSV, perfect steps only",
                    zorder=3, **marker(GREEN, "-.", "^"))
        cl = _lemma_c(C)
        if cl < d.c.max():
            ax.axvspan(cl, d.c.max() + 0.5, color="#eef1f6", zorder=0, lw=0)
        ax.set_xlim(d.c.min() - 0.25, d.c.max() + 0.25)
        ax.set_title(_fmt_C(C) + rf",  $n = {n}$")
        ax.set_xlabel(_c_label())
    axes[0].set_ylabel(r"overlap $|\widehat M \cap M^\ast|/n$")
    h, lab = axes[0].get_legend_handles_labels()
    h.append(plt.Rectangle((0, 0), 1, 1, color="#eef1f6"))
    lab.append(r"$c > 1+\sqrt{6C}$ (Lemma peeling)")
    _legend_below(fig, h, lab, 3, first=r"JSV $\widetilde M(W)$ (all steps)")
    _save(fig, out, "jsv_recovery_vs_c")


def gap_vs_c(df, out, Cs):
    if "jsv_minus_swap" not in df:
        return
    ns = sorted(df.n.unique())
    cols = SEQ[-len(ns):] if len(ns) <= len(SEQ) else [BLUE] * len(ns)
    fig, axes = _panels(len(Cs))
    for ax, C in zip(axes, Cs):
        for n, col in zip(ns, cols):
            s = _s(df[(df.n == n) & np.isclose(df.C, C)], "jsv_minus_swap")
            ax.errorbar(s.c, s.m, yerr=Z * s.se, label=rf"$n = {n}$", **marker(col))
        ax.axhline(0, color=GREY, lw=1.0, ls="--", zorder=0)
        ax.set_title(_fmt_C(C))
        ax.set_xlabel(_c_label())
    axes[0].set_ylabel(r"$\mathrm{ov}(\widetilde M_{\rm JSV}) - \mathrm{ov}(M_{\rm OPT})$")
    h, lab = axes[0].get_legend_handles_labels()
    _legend_below(fig, h, lab, min(len(ns), 6))
    _save(fig, out, "jsv_gap_vs_c")


def occupancy(df, out, n, Cs):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.4, 4.3))
    for C, col in zip(Cs, CYCLE):
        d = df[(df.n == n) & np.isclose(df.C, C)]
        s = _s(d, "frac_perfect")
        a1.errorbar(s.c, s.m, yerr=Z * s.se, label=_fmt_C(C), **marker(col))
        s = _s(d, "mean_size")
        a2.errorbar(s.c, 1 - s.m, yerr=Z * s.se, label=_fmt_C(C), **marker(col))
    a1.set_ylabel(r"fraction of steps with $M_t \in \mathscr{M}_n$")
    a1.set_title(rf"Time on perfect matchings ($n = {n}$)")
    a2.set_ylabel(r"$1 - \mathbb{E}|M_t|/n$  (log scale)")
    a2.set_yscale("log")
    a2.set_title("Fraction of uncovered vertices")
    for a in (a1, a2):
        a.set_xlabel(_c_label())
    h, lab = a1.get_legend_handles_labels()
    _legend_below(fig, h, lab, len(Cs))
    _save(fig, out, "jsv_occupancy")


def diagnostics(df, out, n, Cs):
    if df.q_jsv_empty.isna().all() or df.q_jsv_mle.isna().all():
        return
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.4, 4.3))
    for C, col in zip(Cs, CYCLE):
        d = df[(df.n == n) & np.isclose(df.C, C)].assign(
            g=lambda x: (x.q_jsv_empty - x.q_jsv_mle).abs())
        s = _s(d, "g")
        a1.plot(s.c, s.m.clip(lower=1e-5), label=_fmt_C(C), **marker(col))
        s = _s(d, "jsv_agree")
        a2.errorbar(s.c, s.m, yerr=Z * s.se, label=_fmt_C(C), **marker(col))
    a1.set_yscale("log")
    a1.set_ylabel(r"mean $|q_{\varnothing\text{-start}} - q_{\mathrm{MLE\text{-}start}}|$")
    a1.set_title(rf"Start-point gap ($n = {n}$)")
    a2.set_ylabel(r"$|\widetilde M_{\varnothing} \cap \widetilde M_{\rm MLE}|/n$")
    a2.set_title("Agreement of the two JSV estimates")
    for a in (a1, a2):
        a.set_xlabel(_c_label())
    h, lab = a1.get_legend_handles_labels()
    _legend_below(fig, h, lab, len(Cs))
    _save(fig, out, "jsv_diagnostics")


def make_all_jsv(cfg):
    style.apply()
    out = Path(cfg["out_dir"])
    res = out / "results.csv"
    if not res.exists():
        from .experiment import aggregate
        aggregate(cfg)
    df = pd.read_csv(res)
    p = cfg.get("plots", {})
    n = int(p.get("jsv_n", df.n.max()))
    Cs = [float(c) for c in p.get("jsv_C", sorted(df.C.unique())[:4])]
    figdir = out / "figures"
    print(f"plotting {df.key.nunique()} instances x {df.c.nunique()} penalties -> {figdir}")
    recovery_vs_c(df, figdir, n, Cs)
    gap_vs_c(df, figdir, Cs)
    occupancy(df, figdir, n, Cs)
    diagnostics(df, figdir, n, Cs)

"""Figures in the style of the write-up.

    recovery_vs_C       overlap |M cap M*|/n of Bayes-OPT, MLE and a posterior
                        draw vs C, one panel per n, with the AoN limit
    bayes_advantage     paired difference  ov(Bayes-OPT) - ov(MLE)  vs C
    error_scaling       r_n = 1 - overlap vs n (log-log) for 2 < C, with the
                        n^{-delta(C)} rate of Dai-Cullina-Kiyavash
    qn_scaling          q_n = 1 - mmse_n vs n for C < 2 with the leading term
                        and scalar reference of Prop. mmse-asymptotics
                        (reproduces the write-up's MCMC figure)
    diagnostics         random-start vs MLE-start chains (convergence check)
    recovery_zoom       recovery_vs_C zoomed on the C range where Bayes-OPT and
                        MLE differ, plus the paired difference underneath
    compare_small_vs_large_n
                        Bayes advantage and convergence gap for this run vs
                        the run in plots.compare_with
"""

from __future__ import annotations

import os
from fractions import Fraction
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import style, theory  # noqa: E402
from .style import BLUE, GREY, RUST, SEQ, marker  # noqa: E402

Z = 1.96  # 95% intervals


def _summ(df, col):
    g = df.groupby(["n", "C"])[col]
    return pd.DataFrame({"m": g.mean(), "se": g.sem().fillna(0.0)}).reset_index()


def _fmt_C(C):
    for num, den in ((1, 4), (2, 3), (1, 3), (1, 2), (3, 4)):
        if abs(C - num / den) < 1e-6:
            return rf"$C = \frac{{{num}}}{{{den}}}$"
    return rf"$C = {C:g}$"


def _pow2(n):
    k = np.log2(n)
    return rf"$2^{{{int(k)}}}$" if abs(k - round(k)) < 1e-9 else f"{n}"


def _legend_below(fig, handles, labels, ncol, first=None):
    if first is not None and first in labels:
        k = labels.index(first)
        handles = [handles[k]] + handles[:k] + handles[k + 1:]
        labels = [labels[k]] + labels[:k] + labels[k + 1:]
    fig.tight_layout(rect=(0, 0.11, 1, 1))
    fig.legend(handles, labels, loc="lower center", ncol=ncol,
               bbox_to_anchor=(0.5, 0.0), handlelength=2.6, columnspacing=3)


def _save(fig, out, name):
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}")
    plt.close(fig)
    print(f"  wrote {out / name}.pdf/png")


def _shade_threshold(ax):
    ax.axvline(2.0, color=GREY, lw=0.9, ls=(0, (1, 2)), zorder=0)


# ----------------------------------------------------------------------------
def recovery_vs_C(df, out, ns=None):
    allns = sorted(df.n.unique())
    ns = ns or allns[-3:]
    fig, axes = plt.subplots(1, len(ns), figsize=(5.2 * len(ns), 4.3),
                             sharey=True, squeeze=False)
    for ax, n in zip(axes[0], ns):
        d = df[df.n == n]
        Cg = np.linspace(d.C.min(), d.C.max(), 400)
        ax.plot(Cg, theory.bayes_aon_overlap(Cg), color="#b9bec8", lw=1.2,
                ls=(0, (5, 3)), zorder=1)
        for col, kw, lab in (("ov_bayes", marker(BLUE), "Bayes-OPT (swap chain)"),
                             ("ov_mle", marker(RUST, "--", "s"), "MLE"),
                             ("q", dict(color=GREY, ls=":", lw=1.6),
                              r"posterior draw ($q_n$)")):
            s = _summ(d, col)
            ax.errorbar(s.C, s.m, yerr=Z * s.se, label=lab, zorder=3, **kw)
        _shade_threshold(ax)
        ax.set_title(rf"$n = {n}$")
        ax.set_xlabel(r"$C = \mu^2/\log n$")
        ax.set_ylim(-0.03, 1.03)
    axes[0, 0].set_ylabel(r"overlap $|\widehat M \cap M^\ast|/n$")
    h, lab = axes[0, 0].get_legend_handles_labels()
    h.append(plt.Line2D([], [], color="#b9bec8", lw=1.2, ls=(0, (5, 3))))
    lab.append(r"AoN limit $n\to\infty$")
    _legend_below(fig, h, lab, 4, first="Bayes-OPT (swap chain)")
    _save(fig, out, "recovery_vs_C")


def recovery_zoom(df, out, ns=None, C_range=(1.0, 3.5)):
    """Zoom of recovery_vs_C on the C range where Bayes-OPT and MLE differ.

    Top row: overlaps with 95% CIs, y-axis fitted to the data in the window.
    Bottom row: the paired difference ov(Bayes-OPT) - ov(MLE) (same instances),
    whose CI is much tighter than the two marginal CIs suggest.
    """
    allns = sorted(df.n.unique())
    ns = ns or allns[-3:]
    lo, hi = C_range
    fig, axes = plt.subplots(2, len(ns), figsize=(5.2 * len(ns), 7.4),
                             sharex=True, squeeze=False,
                             gridspec_kw=dict(height_ratios=(3, 2)))
    for k, n in enumerate(ns):
        d = df[(df.n == n) & (df.C >= lo - 1e-9) & (df.C <= hi + 1e-9)]
        ax, axd = axes[0, k], axes[1, k]
        Cg = np.linspace(lo, hi, 300)
        ax.plot(Cg, theory.bayes_aon_overlap(Cg), color="#b9bec8", lw=1.2,
                ls=(0, (5, 3)), zorder=1)
        ylo, yhi = [], []
        for col, kw, lab in (("ov_bayes", marker(BLUE), "Bayes-OPT (swap chain)"),
                             ("ov_mle", marker(RUST, "--", "s"), "MLE")):
            s = _summ(d, col)
            ax.errorbar(s.C, s.m, yerr=Z * s.se, label=lab, zorder=3, capsize=2.5,
                        **kw)
            ylo.append((s.m - Z * s.se).min())
            yhi.append((s.m + Z * s.se).max())
        pad = 0.04 * (max(yhi) - min(ylo))
        ax.set_ylim(min(ylo) - pad, min(1.0, max(yhi)) + pad)
        ax.set_xlim(lo - 0.1, hi + 0.1)
        _shade_threshold(ax)
        ax.set_title(rf"$n = {n}$")
        s = _summ(d.assign(diff=d.ov_bayes - d.ov_mle), "diff")
        axd.fill_between(s.C, s.m - Z * s.se, s.m + Z * s.se, color=BLUE,
                         alpha=0.18, lw=0)
        axd.plot(s.C, s.m, **marker(BLUE))
        axd.axhline(0, color=GREY, lw=1.0, ls="--", zorder=0)
        _shade_threshold(axd)
        axd.set_xlabel(r"$C = \mu^2/\log n$")
    axes[0, 0].set_ylabel(r"overlap $|\widehat M \cap M^\ast|/n$")
    axes[1, 0].set_ylabel(r"$\mathrm{ov}(M_{\mathrm{OPT}}) - \mathrm{ov}(M_{\mathrm{MLE}})$")
    h, lab = axes[0, 0].get_legend_handles_labels()
    h.append(plt.Line2D([], [], color="#b9bec8", lw=1.2, ls=(0, (5, 3))))
    lab.append(r"AoN limit $n\to\infty$")
    h.append(plt.Rectangle((0, 0), 1, 1, color=BLUE, alpha=0.18, lw=0))
    lab.append("paired difference, 95% CI")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.legend(h, lab, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.0),
               handlelength=2.6, columnspacing=3)
    _save(fig, out, "recovery_zoom")


def compare_runs(df_small, df_large, out):
    """Small-n vs large-n runs: Bayes advantage and chain convergence gap vs C."""
    ns_s, ns_l = sorted(df_small.n.unique()), sorted(df_large.n.unique())
    cols_s = plt.cm.Blues(np.linspace(0.45, 0.95, len(ns_s)))
    cols_l = plt.cm.Oranges(np.linspace(0.4, 0.95, len(ns_l)))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.6, 4.6))
    for df, ns, cols, ls, mk in ((df_small, ns_s, cols_s, "-", "o"),
                                 (df_large, ns_l, cols_l, "--", "s")):
        for n, c in zip(ns, cols):
            d = df[df.n == n]
            s = _summ(d.assign(diff=d.ov_bayes - d.ov_mle), "diff")
            a1.errorbar(s.C, s.m, yerr=Z * s.se, color=c, ls=ls, marker=mk,
                        ms=4, lw=1.3, capsize=2, label=rf"$n = {n}$")
            g = _summ(d.assign(g=(d.q_rand - d.q_mle).abs()), "g")
            a2.plot(g.C, g.m.clip(lower=1e-5), color=c, ls=ls, marker=mk, ms=4,
                    lw=1.3, label=rf"$n = {n}$")
    a1.axhline(0, color=GREY, lw=1.0, ls="--", zorder=0)
    for a in (a1, a2):
        _shade_threshold(a)
        a.set_xlabel(r"$C = \mu^2/\log n$")
    a1.set_ylabel(r"$\mathrm{ov}(M_{\mathrm{OPT}}) - \mathrm{ov}(M_{\mathrm{MLE}})$")
    a1.set_title("Bayes-OPT advantage (paired, 95% CI)")
    a2.set_yscale("log")
    a2.set_ylabel(r"mean $|q_{\mathrm{rand}} - q_{\mathrm{MLE}}|$")
    a2.set_title("Convergence gap of the swap chain")
    h, lab = a1.get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.legend(h, lab, loc="lower center", ncol=len(lab), bbox_to_anchor=(0.5, 0.0),
               handlelength=2.2, columnspacing=1.4, fontsize="small")
    _save(fig, out, "compare_small_vs_large_n")


def bayes_advantage(df, out):
    ns = sorted(df.n.unique())
    cols = SEQ[-len(ns):] if len(ns) <= len(SEQ) else plt.cm.Blues(
        np.linspace(0.35, 1, len(ns)))
    fig, ax = plt.subplots(figsize=(6.4, 4.3))
    for n, c in zip(ns, cols):
        s = _summ(df[df.n == n].assign(diff=lambda x: x.ov_bayes - x.ov_mle), "diff")
        ax.errorbar(s.C, s.m, yerr=Z * s.se, label=rf"$n = {n}$", **marker(c))
    ax.axhline(0, color=GREY, lw=1.0, ls="--", zorder=0)
    _shade_threshold(ax)
    ax.set_xlabel(r"$C = \mu^2/\log n$")
    ax.set_ylabel(r"$\mathrm{ov}(M_{\mathrm{OPT}}) - \mathrm{ov}(M_{\mathrm{MLE}})$")
    ax.set_title("Bayes-OPT advantage over the MLE")
    ax.legend(ncol=2, loc="upper left")
    _save(fig, out, "bayes_advantage")


def error_scaling(df, out, Cs=(2.5, 3.0, 3.5)):
    avail = np.array(sorted(df.C.unique()))
    picks = []
    for c in Cs:
        if len(avail) and np.min(np.abs(avail - c)) < 0.13:
            picks.append(float(avail[np.argmin(np.abs(avail - c))]))
    # keep only C values with some nonzero error (log axes)
    picks = sorted(c for c in set(picks)
                   if (df[np.isclose(df.C, c)][["ov_bayes", "ov_mle"]] < 1).any().any())
    if not picks or df.n.nunique() < 2:
        return
    fig, axes = plt.subplots(1, len(picks), figsize=(5.2 * len(picks), 4.3),
                             squeeze=False)
    for ax, C in zip(axes[0], picks):
        d = df[np.isclose(df.C, C)]
        for col, kw, lab in (("ov_bayes", marker(BLUE), "Bayes-OPT"),
                             ("ov_mle", marker(RUST, "--", "s"), "MLE")):
            s = _summ(d, col).sort_values("n")
            r = 1 - s.m
            ok = r > 0
            ax.errorbar(s.n[ok], r[ok], yerr=Z * s.se[ok], label=lab, **kw)
        delta = float(theory.mle_error_exponent([C])[0])
        sm = _summ(d, "ov_mle").sort_values("n")
        r = (1 - sm.m).to_numpy()
        if np.any(r > 0):
            k = np.where(r > 0)[0][-1]
            ng = np.geomspace(sm.n.min(), sm.n.max(), 50)
            ax.plot(ng, r[k] * (ng / sm.n.iloc[k]) ** (-delta), color=GREY,
                    ls=":", lw=1.6, label=rf"$\propto n^{{-\delta(C)}}$")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_title(_fmt_C(C))
        ax.set_xlabel(r"$n$ (log scale)")
    axes[0, 0].set_ylabel(r"$r_n = 1 - |\widehat M\cap M^\ast|/n$ (log scale)")
    h, lab = axes[0, 0].get_legend_handles_labels()
    _legend_below(fig, h, lab, 3, first="Bayes-OPT")
    _save(fig, out, "error_scaling")


def qn_scaling(df, out, Cs=None):
    d0 = df[df.C < 2]
    Cs = ([float(Fraction(str(c))) for c in Cs] if Cs
          else sorted(d0.C.unique())[:3])
    if not Cs or d0.n.nunique() < 3:
        return
    fig, axes = plt.subplots(1, len(Cs), figsize=(5.2 * len(Cs), 4.3),
                             squeeze=False)
    for ax, C in zip(axes[0], Cs):
        s = _summ(d0[np.isclose(d0.C, C)], "q").sort_values("n")
        ng = np.geomspace(s.n.min(), s.n.max(), 60)
        ax.errorbar(s.n, s.m, yerr=Z * s.se, label="MCMC", **marker(BLUE))
        ax.plot(ng, theory.qn_leading(ng, C), color=RUST, ls="--",
                label="Leading term")
        ax.plot(ng, theory.qn_scalar(ng, C), color=GREY, ls=":", lw=1.6,
                label="Scalar")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_title(_fmt_C(C))
        ax.set_xlabel(r"$n$ (log scale)")
    axes[0, 0].set_ylabel(r"$q_n = 1 - \mathsf{mmse}_n$ (log scale)")
    h, lab = axes[0, 0].get_legend_handles_labels()
    _legend_below(fig, h, lab, 3, first="MCMC")
    _save(fig, out, "qn_scaling")


def diagnostics(df, out):
    if df.q_rand.isna().all() or df.q_mle.isna().all():
        return
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.8, 4.4))
    sc = a1.scatter(df.q_mle, df.q_rand, c=df.C, s=10, cmap="viridis",
                    alpha=0.8, linewidths=0)
    a1.plot([0, 1], [0, 1], color=GREY, lw=1, ls="--")
    a1.set_xlabel(r"$q$ from MLE-started chains")
    a1.set_ylabel(r"$q$ from random-started chains")
    a1.set_title("Start-point agreement (per instance)")
    fig.colorbar(sc, ax=a1, label=r"$C$")
    ns = sorted(df.n.unique())
    cols = SEQ[-len(ns):] if len(ns) <= len(SEQ) else [BLUE] * len(ns)
    for n, c in zip(ns, cols):
        s = _summ(df[df.n == n].assign(g=lambda x: (x.q_rand - x.q_mle).abs()), "g")
        a2.plot(s.C, s.m.clip(lower=1e-5), label=rf"$n={n}$", **marker(c))
    _shade_threshold(a2)
    a2.set_yscale("log")
    a2.set_xlabel(r"$C$")
    a2.set_ylabel(r"mean $|q_{\mathrm{rand}} - q_{\mathrm{MLE}}|$")
    a2.set_title("Convergence gap")
    a2.legend(ncol=2)
    _save(fig, out, "diagnostics")


def make_all(cfg):
    style.apply()
    out = Path(cfg["out_dir"])
    res = out / "results.csv"
    if not res.exists():
        from .experiment import aggregate
        aggregate(cfg)
    df = pd.read_csv(res)
    figdir = out / "figures"
    p = cfg.get("plots", {})
    print(f"plotting {len(df)} instances -> {figdir}")
    if df.C.nunique() >= 3:
        recovery_vs_C(df, figdir, p.get("recovery_n"))
        recovery_zoom(df, figdir, p.get("recovery_n"),
                      tuple(p.get("zoom_C", (1.0, 3.5))))
        bayes_advantage(df, figdir)
        other = p.get("compare_with")
        if other:
            other = Path(os.path.expandvars(other))
            if other.exists():
                compare_runs(df, pd.read_csv(other), figdir)
            else:
                print(f"  compare_with: {other} not found, skipping")
        diagnostics(df, figdir)
    error_scaling(df, figdir, tuple(p.get("scaling_C", (2.5, 3.0, 3.5))))
    qn_scaling(df, figdir, p.get("qn_C"))

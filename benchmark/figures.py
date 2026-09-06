"""Figures built from the CSVs in ``results/``.

Each function reads a results file and writes a PNG into ``figures/``. They are
deliberately independent of the run itself, so figures can be regenerated
without recomputing anything.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

__all__ = ["fig_agreement", "fig_hq_curves", "fig_invariance", "fig_prevalence",
           "fig_cross_corpus", "build_all"]

# One stable colour per implementation so figures are comparable.
COLORS = {
    "MFDFA": "#d62728",
    "fathon": "#ff7f0e",
    "fdnkit": "#2ca02c",
    "fdnkit_unguarded": "#8c564b",
    "neurokit2": "#1f77b4",
    "nolds": "#7f7f7f",
}


def _save(fig, name, dpi=600):
    """Write the figure twice: a raster at print resolution and a vector copy.

    These are line plots, so the PDF is the version to send a publisher; it has no resolution to
    be wrong. The PNG exists because Markdown and Word embed it, and it is written at 600 dpi
    because journals commonly require 600 to 1200 for line art and reject 150.
    """
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / name
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    return path


def fig_agreement(summary_csv="summary.csv", name="fig1_agreement.png"):
    """delta_h per implementation, synthetic controls versus real EEG.

    The point of the figure is the contrast: near-identical on signals whose
    answer is known, wildly different on real recordings.
    """
    df = pd.read_csv(RESULTS / summary_csv)
    df = df[df.package != "nolds"]
    order = [s for s in df.signal.unique()]
    is_real = [s.startswith("eegbci") for s in order]
    # Group synthetic first, real last, for a readable left-to-right story.
    order = [s for s, r in zip(order, is_real) if not r] + \
            [s for s, r in zip(order, is_real) if r]

    fig, ax = plt.subplots(figsize=(11, 5))
    pkgs = sorted(df.package.unique())
    width = 0.8 / len(pkgs)
    xs = np.arange(len(order))
    for k, pkg in enumerate(pkgs):
        vals = [df[(df.signal == s) & (df.package == pkg)].delta_h.mean() for s in order]
        ax.bar(xs + k * width, vals, width, label=pkg,
               color=COLORS.get(pkg, None), edgecolor="none")
    n_syn = sum(1 for s in order if not s.startswith("eegbci"))
    if 0 < n_syn < len(order):
        ax.axvline(n_syn - 0.15, color="0.3", ls="--", lw=1)
        ax.text(n_syn - 0.1, ax.get_ylim()[1] * 0.95, " real EEG", va="top", fontsize=9)
    ax.set_yscale("symlog", linthresh=0.1)
    ax.set_xticks(xs + 0.4 - width / 2)
    ax.set_xticklabels(order, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel(r"multifractal width $\Delta h$  (symlog)")
    ax.set_title("Implementations agree on signals with known answers, and diverge on real EEG")
    ax.legend(fontsize=8, ncol=len(pkgs))
    ax.grid(axis="y", alpha=0.3)
    return _save(fig, name)


def fig_hq_curves(long_csv="hq_long.csv", signal_contains="eegbci",
                  name="fig2_hq_curves.png"):
    """h(q) against q for one real channel and one synthetic control."""
    df = pd.read_csv(RESULTS / long_csv)
    real = [s for s in df.signal.unique() if signal_contains in s]
    syn = [s for s in df.signal.unique() if s.startswith("fgn_H0.7") and "flat" not in s]
    picks = [(syn[0] if syn else df.signal.iloc[0], "synthetic control, true h(q) = 0.7"),
             (real[0] if real else df.signal.iloc[-1], "real EEG channel")]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True)
    for ax, (sig, title) in zip(axes, picks):
        sub = df[df.signal == sig]
        for pkg, s in sub.groupby("package"):
            if pkg == "nolds" or not np.isfinite(s.hq).any():
                continue
            s = s.sort_values("q")
            ax.plot(s.q, s.hq, "o-", ms=4, label=pkg, color=COLORS.get(pkg, None))
        truth = sub.dropna(subset=["hq_truth"])
        if not truth.empty:
            t = truth.drop_duplicates("q").sort_values("q")
            ax.plot(t.q, t.hq_truth, "k--", lw=1.5, label="analytic truth")
        ax.set_title(f"{title}\n{sig}", fontsize=9)
        ax.set_xlabel("q")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("h(q)")
    axes[1].legend(fontsize=8)
    return _save(fig, name)


def fig_invariance(inv_csv="inv_invariance.csv", name="fig3_invariance.png"):
    """Drift of h(q) as the input is rescaled. A correct implementation is flat at 0."""
    path = RESULTS / inv_csv
    if not path.exists():
        return None
    df = pd.read_csv(path)
    real = [s for s in df.signal.unique() if s.startswith("eegbci")]
    sig = real[0] if real else df.signal.iloc[0]
    sub = df[df.signal == sig]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for pkg, s in sub.groupby("package"):
        s = s.sort_values("multiplier")
        ax.plot(s.multiplier, s.max_abs_drift, "o-", ms=5, label=pkg,
                color=COLORS.get(pkg, None))
    # Mark outright failures explicitly. Without this a package that errored is
    # simply absent from the line, which reads as "no drift" rather than "worse".
    failed = sub[sub.failed]
    if not failed.empty:
        ymax = ax.get_ylim()[1]
        ax.scatter(failed.multiplier, [ymax] * len(failed), marker="x", s=70,
                   c=[COLORS.get(p, "k") for p in failed.package], zorder=5)
        ax.text(failed.multiplier.min(), ymax, " x = returned no result ",
                fontsize=8, va="bottom", ha="left")
    ax.set_xscale("log")
    ax.set_yscale("symlog", linthresh=1e-3)
    ax.axhline(0, color="k", lw=1)
    ax.set_xlabel("multiplier applied to the signal")
    ax.set_ylabel(r"max$_q$ |h(q) drift| vs unscaled")
    ax.set_title(f"h(q) must not depend on the units of the input\n{sig}", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return _save(fig, name)


def fig_prevalence(prev_csv="prevalence.csv", name="fig4_prevalence.png"):
    """Cohort distribution of delta_h, and its relation to flat-run content."""
    path = RESULTS / prev_csv
    if not path.exists():
        return None
    df = pd.read_csv(path)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    pkgs = sorted(df.package.unique())
    data = [df[df.package == p].delta_h.replace([np.inf, -np.inf], np.nan).dropna()
            for p in pkgs]
    # matplotlib >= 3.9 renamed 'labels' to 'tick_labels'; support both.
    try:
        axes[0].boxplot(data, tick_labels=pkgs, showfliers=False)
    except TypeError:
        axes[0].boxplot(data, labels=pkgs, showfliers=False)
    axes[0].set_yscale("symlog", linthresh=0.1)
    axes[0].set_ylabel(r"$\Delta h$ (symlog)")
    axes[0].set_title("Distribution across the cohort", fontsize=10)
    axes[0].tick_params(axis="x", rotation=25, labelsize=8)
    axes[0].grid(axis="y", alpha=0.3)

    # Flat content is effectively bimodal (a channel either has constant runs or
    # it does not), so a categorical split is far more legible than a scatter.
    ax = axes[1]
    for k, pkg in enumerate(pkgs):
        s = df[df.package == pkg].dropna(subset=["delta_h"])
        for j, (lab, mask) in enumerate((("no flat runs", s.flat_runs == 0),
                                         ("has flat runs", s.flat_runs > 0))):
            v = s[mask].delta_h.values
            if not len(v):
                continue
            xpos = k + (j - 0.5) * 0.32
            ax.scatter(np.full(len(v), xpos) + np.random.default_rng(0).normal(0, .03, len(v)),
                       v, s=12, alpha=0.55,
                       color=COLORS.get(pkg, None), marker="o" if j == 0 else "^")
    ax.axhline(1.0, color="0.4", ls="--", lw=1)
    ax.text(len(pkgs) - 0.5, 1.05, "implausible for EEG", fontsize=7, ha="right", va="bottom")
    ax.set_xticks(range(len(pkgs)))
    ax.set_xticklabels(pkgs, rotation=25, fontsize=8)
    ax.set_ylabel(r"$\Delta h$")
    ax.set_yscale("symlog", linthresh=0.1)
    ax.set_title("circles = channel has no flat runs; triangles = channel has flat runs",
                 fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    return _save(fig, name)


def fig_cross_corpus(name="fig5_cross_corpus.png"):
    """Does the effect replicate on a second corpus recorded on other hardware?

    Reads both prevalence tables and shows the delta_h distribution per
    implementation, split by corpus and by whether the channel contains a
    constant run.
    """
    a = RESULTS / "prevalence.csv"
    b = RESULTS / "prevalence_sleepedf.csv"
    if not (a.exists() and b.exists()):
        return None
    da = pd.read_csv(a); da["corpus"] = "EEGBCI (160 Hz)"
    db = pd.read_csv(b); db["corpus"] = "Sleep-EDF (100 Hz)"
    df = pd.concat([da, db], ignore_index=True).dropna(subset=["delta_h"])

    pkgs = sorted(df.package.unique())
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
    for ax, corpus in zip(axes, df.corpus.unique()):
        sub = df[df.corpus == corpus]
        for k, pkg in enumerate(pkgs):
            s = sub[sub.package == pkg]
            for j, mask in enumerate((s.flat_runs == 0, s.flat_runs > 0)):
                v = s[mask].delta_h.values
                if not len(v):
                    continue
                xpos = k + (j - 0.5) * 0.32
                jitter = np.random.default_rng(0).normal(0, 0.03, len(v))
                ax.scatter(np.full(len(v), xpos) + jitter, v, s=12, alpha=0.55,
                           color=COLORS.get(pkg, None), marker="o" if j == 0 else "^")
        ax.axhline(1.0, color="0.4", ls="--", lw=1)
        # Annotate how many channels each package actually returned a number for.
        # Without this, a package that failed on most of the corpus simply looks
        # sparse, which reads as "few outliers" rather than "mostly unusable".
        raw = pd.read_csv(a if "EEGBCI" in corpus else b)
        for k, pkg in enumerate(pkgs):
            r = raw[raw.package == pkg]
            nan_pct = 100.0 * (~np.isfinite(r.delta_h)).mean() if len(r) else 0.0
            if nan_pct > 0:
                ax.annotate(f"{nan_pct:.0f}% NaN", xy=(k, 0.012), ha="center",
                            fontsize=7, color="firebrick")
        ax.set_xticks(range(len(pkgs)))
        ax.set_xticklabels(pkgs, rotation=25, fontsize=8)
        ax.set_yscale("symlog", linthresh=0.1)
        ax.set_title(corpus, fontsize=10)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel(r"$\Delta h$ (symlog)")
    fig.suptitle("circles = no constant runs, triangles = channel contains constant runs",
                 fontsize=9)
    return _save(fig, name)


def build_all():
    made = []
    for fn in (fig_agreement, fig_hq_curves, fig_invariance, fig_prevalence,
               fig_cross_corpus):
        try:
            p = fn()
            if p:
                made.append(p)
                print(f"  wrote {p.name}")
        except Exception as exc:
            print(f"  {fn.__name__} skipped: {type(exc).__name__}: {exc}")
    return made


if __name__ == "__main__":
    build_all()

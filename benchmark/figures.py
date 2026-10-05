"""Figures built from the CSVs in ``results/``.

Each function reads a results file and writes a PNG and a PDF into ``figures/``. They are
deliberately independent of the run itself, so figures can be regenerated without recomputing
anything.

Layout follows the journal's figure guidance: figures are drawn at their printed width (15 cm for a
full-width figure) so that text is 8 to 9 pt as printed, the parts of a multi-panel figure carry a
lower-case letter in parentheses, legends sit outside the plotting area, and implementations are
told apart by marker shape as well as colour so the figures survive greyscale printing. Plot titles
name what is shown; the captions carry the interpretation. Files are numbered in the order the
figures are cited in the manuscript.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

__all__ = ["fig_agreement", "fig_hq_curves", "fig_prevalence", "fig_invariance",
           "fig_cross_corpus", "build_all"]

FULL_WIDTH = 15 / 2.54  # inches; a full-width journal figure
plt.rcParams.update({
    "font.size": 8.5, "axes.titlesize": 8.5, "axes.labelsize": 8.5, "xtick.labelsize": 8,
    "ytick.labelsize": 8, "legend.fontsize": 8, "font.family": "DejaVu Sans",
})

# One stable colour and marker per implementation so figures are comparable.
COLORS = {
    "MFDFA": "#d62728",
    "fathon": "#ff7f0e",
    "fdnkit": "#2ca02c",
    "fdnkit_unguarded": "#8c564b",
    "neurokit2": "#1f77b4",
    "nolds": "#7f7f7f",
}
MARKERS = {"MFDFA": "o", "fathon": "s", "fdnkit": "D", "fdnkit_unguarded": "v",
           "neurokit2": "^", "nolds": "P"}
HATCHES = {"MFDFA": "", "fathon": "///", "fdnkit": "", "fdnkit_unguarded": "xxx",
           "neurokit2": "...", "nolds": ""}
# Display names match the manuscript's tables.
LABELS = {"fdnkit_unguarded": "fdnkit, floor off"}


def _label(pkg):
    return LABELS.get(pkg, pkg)


def _panel(ax, letter, title=""):
    """Lower-case panel letter in parentheses, left-aligned above the axes, followed by the
    panel's title, so the two can never overprint each other."""
    ax.set_title(f"({letter})  {title}", loc="left", fontsize=8.5)


def _legend_below(fig, handles, labels, ncol, y=0.0):
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, y), ncol=ncol,
               frameon=False, handletextpad=0.4, columnspacing=1.2)


def _save(fig, name, dpi=600):
    """Write the figure twice: a raster at print resolution and a vector copy.

    These are line plots, so the PDF is the version to send a publisher; it has no resolution to
    be wrong. The PNG exists because Markdown and Word embed it, and it is written at 600 dpi
    because journals commonly require 600 to 1200 for line art and reject 150. The bounding box is
    tight so that a legend placed below the axes is kept.
    """
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / name
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    return path


SIGNAL_LABELS = {
    "fgn_H0.3": "fGn H = 0.3", "fgn_H0.5": "fGn H = 0.5", "fgn_H0.7": "fGn H = 0.7",
    "fgn_H0.9": "fGn H = 0.9", "white_noise": "white noise", "cascade_p0.3": "cascade",
    "fgn_H0.7_flat1x32": "1 run of 32", "fgn_H0.7_flat5x32": "5 runs of 32",
    "fgn_H0.7_flat20x64": "20 runs of 64",
}


def _signal_label(s):
    if s.startswith("eegbci_"):
        return "channel " + s.rsplit("_", 1)[-1]
    return SIGNAL_LABELS.get(s, s)


def fig_agreement(summary_csv="summary.csv", name="fig1_agreement.png"):
    """delta_h per implementation in three groups: clean controls, controls carrying constant
    runs, and recorded EEG.

    The constant-run controls are kept apart from the clean controls: their true h(q) is known,
    but the implementations do not agree on them, so grouping them with the clean controls would
    contradict the point the figure makes.
    """
    df = pd.read_csv(RESULTS / summary_csv)
    df = df[df.package != "nolds"]
    signals = list(df.signal.unique())
    groups = [("controls,\nknown exponents",
               [s for s in signals if not s.startswith("eegbci") and "flat" not in s]),
              ("controls with\nconstant runs", [s for s in signals if "flat" in s]),
              ("recorded EEG,\nmotor imagery", [s for s in signals if s.startswith("eegbci")])]
    order = [s for _, g in groups for s in g]

    fig, ax = plt.subplots(figsize=(FULL_WIDTH, 3.0))
    pkgs = sorted(df.package.unique())
    width = 0.8 / len(pkgs)
    xs = np.arange(len(order))
    for k, pkg in enumerate(pkgs):
        vals = [df[(df.signal == s) & (df.package == pkg)].delta_h.mean() for s in order]
        ax.bar(xs + (k - (len(pkgs) - 1) / 2) * width, vals, width, label=_label(pkg),
               color=COLORS.get(pkg), hatch=HATCHES.get(pkg, ""), edgecolor="white", lw=0.3)
    ax.set_yscale("symlog", linthresh=0.1)
    top = ax.get_ylim()[1]
    ax.set_ylim(top=top * 12)
    start = 0
    for i, (title, g) in enumerate(groups):
        if i:
            ax.axvline(start - 0.5, color="0.3", ls="--", lw=0.8)
        ax.text(start - 0.4, top * 5, title, fontsize=8, va="center", ha="left")
        start += len(g)
    ax.set_xticks(xs)
    ax.set_xticklabels([_signal_label(s) for s in order], rotation=40, ha="right")
    ax.set_xlim(-0.6, len(order) - 0.4)
    ax.set_ylabel(r"multifractal width $\Delta h$")
    ax.grid(axis="y", alpha=0.3)
    h, lab = ax.get_legend_handles_labels()
    fig.tight_layout()
    _legend_below(fig, h, lab, ncol=len(pkgs), y=0.0)
    return _save(fig, name)


def fig_hq_curves(long_csv="hq_long.csv", signal_contains="eegbci",
                  name="fig2_hq_curves.png"):
    """h(q) against q for one synthetic control (a) and one recorded channel (b)."""
    df = pd.read_csv(RESULTS / long_csv)
    real = [s for s in df.signal.unique() if signal_contains in s]
    syn = [s for s in df.signal.unique() if s.startswith("fgn_H0.7") and "flat" not in s]
    sig_syn = syn[0] if syn else df.signal.iloc[0]
    sig_real = real[0] if real else df.signal.iloc[-1]
    picks = [(sig_syn, "fGn, H = 0.7"),
             (sig_real, "motor-imagery EEG, " + _signal_label(sig_real))]

    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH, 2.6), sharex=True)
    handles = {}
    for ax, letter, (sig, title) in zip(axes, "ab", picks):
        sub = df[df.signal == sig]
        for pkg, s in sub.groupby("package"):
            if pkg == "nolds" or not np.isfinite(s.hq).any():
                continue
            s = s.sort_values("q")
            (ln,) = ax.plot(s.q, s.hq, marker=MARKERS.get(pkg, "o"), ms=3.5, lw=1,
                            color=COLORS.get(pkg))
            handles[_label(pkg)] = ln
        truth = sub.dropna(subset=["hq_truth"])
        if not truth.empty:
            t = truth.drop_duplicates("q").sort_values("q")
            (ln,) = ax.plot(t.q, t.hq_truth, "k--", lw=1.2)
            handles["analytic value"] = ln
        ax.set_xlabel("moment order q")
        ax.grid(alpha=0.3)
        _panel(ax, letter, title)
    axes[0].set_ylabel("h(q)")
    fig.tight_layout()
    _legend_below(fig, list(handles.values()), list(handles), ncol=len(handles), y=0.0)
    return _save(fig, name)


def _split_markers():
    return [Line2D([], [], ls="", marker="o", color="0.35", ms=4),
            Line2D([], [], ls="", marker="^", color="0.35", ms=4)]


def _split_scatter(ax, sub, pkgs):
    """Per implementation, channels without a constant run (circles, left) and with (triangles,
    right). Constant-run content is effectively bimodal, so a categorical split reads better than
    a scatter against the fraction."""
    rng = np.random.default_rng(0)
    for k, pkg in enumerate(pkgs):
        s = sub[sub.package == pkg]
        for j, mask in enumerate((s.flat_runs == 0, s.flat_runs > 0)):
            v = s[mask].delta_h.values
            if not len(v):
                continue
            xpos = k + (j - 0.5) * 0.34
            ax.scatter(np.full(len(v), xpos) + rng.normal(0, 0.03, len(v)), v, s=9, alpha=0.6,
                       color=COLORS.get(pkg), marker="o" if j == 0 else "^", lw=0)
    ax.axhline(1.0, color="0.4", ls="--", lw=0.8)
    ax.set_xticks(range(len(pkgs)))
    ax.set_xticklabels([_label(p) for p in pkgs], rotation=30, ha="right")
    ax.set_yscale("symlog", linthresh=0.1)
    ax.grid(axis="y", alpha=0.3)


def fig_prevalence(prev_csv="prevalence.csv", name="fig3_prevalence.png"):
    """Motor-imagery corpus: distribution of delta_h (a), and the same values split by whether the
    channel contains a constant run (b)."""
    path = RESULTS / prev_csv
    if not path.exists():
        return None
    df = pd.read_csv(path)
    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH, 2.8), sharey=True)

    pkgs = sorted(df.package.unique())
    data = [df[df.package == p].delta_h.replace([np.inf, -np.inf], np.nan).dropna()
            for p in pkgs]
    # matplotlib >= 3.9 renamed 'labels' to 'tick_labels'; support both.
    try:
        axes[0].boxplot(data, tick_labels=[_label(p) for p in pkgs], showfliers=False)
    except TypeError:
        axes[0].boxplot(data, labels=[_label(p) for p in pkgs], showfliers=False)
    axes[0].set_yscale("symlog", linthresh=0.1)
    axes[0].set_ylabel(r"multifractal width $\Delta h$")
    axes[0].tick_params(axis="x", rotation=30)
    for t in axes[0].get_xticklabels():
        t.set_ha("right")
    axes[0].axhline(1.0, color="0.4", ls="--", lw=0.8)
    axes[0].grid(axis="y", alpha=0.3)
    _split_scatter(axes[1], df.dropna(subset=["delta_h"]), pkgs)
    _panel(axes[0], "a", "all 80 channels")
    _panel(axes[1], "b", "split by constant-run content")
    fig.tight_layout()
    _legend_below(fig, _split_markers(), ["no constant run", "contains a constant run"], ncol=2)
    return _save(fig, name)


def fig_invariance(inv_csv="inv_invariance.csv", name="fig4_invariance.png"):
    """Drift of h(q) as the input is rescaled. A correct implementation is flat at 0."""
    path = RESULTS / inv_csv
    if not path.exists():
        return None
    df = pd.read_csv(path)
    real = [s for s in df.signal.unique() if s.startswith("eegbci")]
    sig = real[0] if real else df.signal.iloc[0]
    sub = df[df.signal == sig]

    fig, ax = plt.subplots(figsize=(FULL_WIDTH * 0.72, 2.9))
    for pkg, s in sub.groupby("package"):
        s = s.sort_values("multiplier")
        ax.plot(s.multiplier, s.max_abs_drift, marker=MARKERS.get(pkg, "o"), ms=4, lw=1,
                label=_label(pkg), color=COLORS.get(pkg))
    ax.set_xscale("log")
    ax.set_yscale("symlog", linthresh=1e-3)
    # Mark outright failures explicitly. Without this a package that errored is simply absent from
    # the line, which reads as "no drift" rather than "worse".
    failed = sub[sub.failed]
    handles, labels = ax.get_legend_handles_labels()
    if not failed.empty:
        ymax = ax.get_ylim()[1]
        ax.scatter(failed.multiplier, [ymax] * len(failed), marker="x", s=40,
                   c=[COLORS.get(p, "k") for p in failed.package], zorder=5, clip_on=False)
        handles.append(Line2D([], [], ls="", marker="x", color="0.2", ms=5))
        labels.append("no result returned")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("multiplier applied to the signal")
    ax.set_ylabel(r"max$_q$ |drift of h(q)|")
    ax.set_title("motor-imagery EEG, " + _signal_label(sig), loc="left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _legend_below(fig, handles, labels, ncol=4)
    return _save(fig, name)


def fig_cross_corpus(name="fig5_cross_corpus.png"):
    """Does the effect replicate on a second corpus recorded on other hardware?

    Reads both prevalence tables and shows the delta_h values per implementation, split by
    corpus, (a) motor imagery and (b) Sleep-EDF, and by whether the channel contains a constant
    run.
    """
    a = RESULTS / "prevalence.csv"
    b = RESULTS / "prevalence_sleepedf.csv"
    if not (a.exists() and b.exists()):
        return None
    corpora = [("motor imagery, 160 Hz", pd.read_csv(a)), ("Sleep-EDF, 100 Hz", pd.read_csv(b))]
    pkgs = sorted(set(corpora[0][1].package) | set(corpora[1][1].package))
    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH, 2.8), sharey=True)
    for ax, letter, (title, raw) in zip(axes, "ab", corpora):
        _split_scatter(ax, raw.dropna(subset=["delta_h"]), pkgs)
        # Annotate how many channels each package returned no number for. Without this, a package
        # that failed on most of the corpus simply looks sparse, which reads as "few outliers"
        # rather than "mostly unusable".
        for k, pkg in enumerate(pkgs):
            r = raw[raw.package == pkg]
            nan_pct = 100.0 * (~np.isfinite(r.delta_h)).mean() if len(r) else 0.0
            if nan_pct > 0:
                ax.annotate(f"{nan_pct:.0f}% no result", xy=(k, 0.012), ha="center",
                            fontsize=7.5, color="firebrick")
        _panel(ax, letter, title)
    axes[0].set_ylabel(r"multifractal width $\Delta h$")
    fig.tight_layout()
    _legend_below(fig, _split_markers(), ["no constant run", "contains a constant run"], ncol=2)
    return _save(fig, name)


def build_all():
    made = []
    for fn in (fig_agreement, fig_hq_curves, fig_prevalence, fig_invariance, fig_cross_corpus):
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

"""Measure how many segments `neurokit2`'s absolute variance threshold removes.

The threshold is `var > 1e-08` in `_fractal_dfa_fluctuation`. Detrended variance carries the
square of the signal's units, so on volt-scale EEG the filter removes legitimate segments, and it
does so unevenly across scales because segment variance grows with window size. This module
measures that directly.

Nothing here reimplements the estimator. `neurokit2._fractal_dfa_fluctuation` is wrapped in its own
module namespace, so the counts come from the same call `fractal_dfa` makes, with whatever
segmentation, integration and detrending the installed version actually uses. Reimplementing the
segmentation gives materially different answers, mostly because `overlap=True` is the default and
the windows therefore stride by `window // 2`, which is exactly the kind of divergence this paper
is about.

The counts are keyed on the window length, read from the segment matrix, so a scale grid change
needs no edit here.
"""
from __future__ import annotations

import contextlib
import statistics
import warnings

import numpy as np


@contextlib.contextmanager
def _counting(counts):
    """Wrap the fluctuation function so each call records (window, total, kept)."""
    import importlib

    # `from neurokit2.complexity import fractal_dfa` yields the re-exported function, not the
    # module that holds the helper, so ask importlib for the module itself.
    mod = importlib.import_module("neurokit2.complexity.fractal_dfa")

    original = mod._fractal_dfa_fluctuation

    def wrapper(segments, trends, q=2):
        window = int(np.asarray(segments).shape[1])
        var = np.var(np.asarray(segments) - np.asarray(trends), axis=1)
        counts.append((window, int(var.size), int(np.count_nonzero(var > 1e-08))))
        return original(segments, trends, q)

    mod._fractal_dfa_fluctuation = wrapper
    try:
        yield
    finally:
        mod._fractal_dfa_fluctuation = original


def discarded_per_scale(signal, scales, q):
    """Fraction of segments removed at each scale, plus whether the call returned no result.

    Returns ``(fractions, failed)`` where ``fractions`` maps window length to the fraction
    removed. A scale absent from the mapping was never reached.
    """
    import neurokit2 as nk

    counts: list[tuple[int, int, int]] = []
    failed = False
    with _counting(counts):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                _, info = nk.fractal_dfa(
                    np.asarray(signal, float), scale=np.asarray(scales, int),
                    q=np.asarray(q, float), multifractal=True, show=False,
                )
                hq = np.asarray(info["h"], dtype=float).ravel()
                failed = hq.size == 0 or bool(np.all(~np.isfinite(hq)))
            except Exception:
                failed = True

    totals: dict[int, list[int]] = {}
    kept: dict[int, list[int]] = {}
    for window, total, keep in counts:
        totals.setdefault(window, []).append(total)
        kept.setdefault(window, []).append(keep)
    fractions = {w: 1.0 - (sum(kept[w]) / sum(totals[w])) if sum(totals[w]) else float("nan")
                 for w in totals}
    return fractions, failed


def survey(pairs, scales, q, report_scales=(16, 64, 256)):
    """Run the measurement over ``(name, signal)`` pairs and summarise one corpus.

    The per-signal rows are returned alongside the summary so the table can be rebuilt from the
    CSV without rerunning, and so a reader can see the spread behind each mean.
    """
    rows, sds = [], []
    for name, sig in pairs:
        sig = np.asarray(sig, float)
        sds.append(float(np.std(sig)))
        fractions, failed = discarded_per_scale(sig, scales, q)
        row = {"signal": name, "signal_sd": float(np.std(sig)), "failed": failed}
        for w in report_scales:
            row[f"discarded_{w}"] = fractions.get(w, float("nan"))
        rows.append(row)

    summary = {
        "n": len(rows),
        "median_signal_sd": statistics.median(sds) if sds else float("nan"),
        "failure_rate": (sum(1 for r in rows if r["failed"]) / len(rows)) if rows else float("nan"),
    }
    for w in report_scales:
        vals = [r[f"discarded_{w}"] for r in rows if not np.isnan(r[f"discarded_{w}"])]
        summary[f"discarded_{w}"] = statistics.fmean(vals) if vals else float("nan")
    return rows, summary

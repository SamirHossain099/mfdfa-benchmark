"""Cohort-scale prevalence run.

The pilot showed large disagreement on a handful of channels. This asks how
often it happens across a cohort, and whether it is predicted by the amount of
constant-run content in each channel, which is the suspected trigger.

For every channel we record each implementation's ``delta_h`` alongside the
channel's flat-run fraction, so the association can be tested directly rather
than assumed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["prevalence_table", "summarize_prevalence"]

# A multifractal width this large is not physically plausible for EEG, where
# published values sit well below 1. Used only for reporting a rate.
IMPLAUSIBLE_DELTA_H = 1.0


def _width(hq) -> float:
    h = np.asarray(hq, dtype=float)
    h = h[np.isfinite(h)]
    return float(h.max() - h.min()) if h.size else float("nan")


def prevalence_table(signals, adapters, scales, q, verbose=True) -> pd.DataFrame:
    """One row per (channel, package) with delta_h and the channel's flat content."""
    from fdnkit.preprocessing import flat_fraction, find_flat_runs

    rows = []
    for i, entry in enumerate(signals):
        name, x = entry[0], np.asarray(entry[1], dtype=float)
        min_scale = int(np.min(scales))
        runs = find_flat_runs(x, min_length=min_scale)
        frac = flat_fraction(x, min_length=min_scale)
        for pkg, fn in adapters.items():
            try:
                hq = np.asarray(fn(x, scales, q).get("hq"), dtype=float).ravel()
                dh = _width(hq) if hq.size == q.size else np.nan
                err = None
            except Exception as exc:
                dh, err = np.nan, type(exc).__name__
            rows.append({
                "signal": name, "package": pkg, "delta_h": dh,
                "flat_runs": int(len(runs)), "flat_fraction": float(frac),
                "error": err,
            })
        if verbose and (i + 1) % 5 == 0:
            print(f"    {i + 1}/{len(signals)} channels", flush=True)
    return pd.DataFrame(rows)


def summarize_prevalence(df: pd.DataFrame, threshold: float = IMPLAUSIBLE_DELTA_H):
    """Per-package rates and the association with flat-run content."""
    out = {}
    g = df.groupby("package")["delta_h"]
    out["by_package"] = pd.DataFrame({
        "median_delta_h": g.median(),
        "max_delta_h": g.max(),
        f"pct_over_{threshold:g}": g.apply(lambda s: 100.0 * np.mean(s > threshold)),
        "pct_nan": g.apply(lambda s: 100.0 * np.mean(~np.isfinite(s))),
    }).round(3)

    # Does flat content predict an implausible width? Spearman avoids assuming
    # linearity, which we have no reason to expect.
    rows = []
    for pkg, sub in df.groupby("package"):
        s = sub.dropna(subset=["delta_h"])
        if len(s) > 3 and s.flat_fraction.nunique() > 1:
            rho = s[["flat_fraction", "delta_h"]].corr(method="spearman").iloc[0, 1]
        else:
            rho = np.nan
        rows.append({"package": pkg, "spearman_flat_vs_delta_h": round(float(rho), 3),
                     "n": len(s)})
    out["association"] = pd.DataFrame(rows).set_index("package")
    return out

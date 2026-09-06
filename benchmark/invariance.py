"""Scale-invariance test.

The generalized Hurst exponent is defined through the scaling of fluctuations
with window size, so multiplying a signal by a positive constant must leave
``h(q)`` unchanged. Expressing an EEG channel in volts or in microvolts is
exactly such a rescaling.

This module measures, for each implementation, how far ``h(q)`` drifts as the
input is rescaled over many orders of magnitude. Any non-zero drift is a defect:
it means the reported exponent depends on the unit the data happens to be stored
in. It is worth testing across packages rather than assuming, because a guard
against near-zero-variance segments is a good idea and only becomes a problem
when its threshold is absolute rather than relative to the data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["MULTIPLIERS", "invariance_table"]

MULTIPLIERS = (1e6, 1e3, 1e0, 1e-3, 1e-4, 1e-5, 1e-6)


def _hq(fn, x, scales, q):
    """Exponents for one call.

    Monofractal-only packages (``nolds``) return no ``h(q)``; for those we fall
    back to the single exponent ``H`` they do report, so the invariance test is
    applied to the quantity each package actually produces rather than marking
    them as failures for a capability they never claimed.
    """
    try:
        res = fn(x, scales, q)
        hq = np.asarray(res.get("hq"), dtype=float).ravel()
        if hq.size == q.size and np.isfinite(hq).any():
            return hq, None
        if res.get("H") is not None and np.isfinite(res["H"]):
            return np.full(q.size, float(res["H"])), None
        return np.full(q.size, np.nan), "no finite output"
    except Exception as exc:
        return np.full(q.size, np.nan), f"{type(exc).__name__}"


def invariance_table(signals, adapters, scales, q, multipliers=MULTIPLIERS,
                     reference: float = 1.0, verbose: bool = True) -> pd.DataFrame:
    """Drift of ``h(q)`` under rescaling, per (signal, package, multiplier).

    Parameters
    ----------
    signals : iterable of ``(name, x)`` or ``(name, x, truth_fn)``
    adapters : mapping name -> ``estimate(x, scales, q)``
    scales, q : array-like
        Shared analysis grid.
    multipliers : iterable of float
        Positive constants the signal is multiplied by.
    reference : float
        The multiplier treated as the baseline for the drift comparison.

    Returns
    -------
    pandas.DataFrame
        Columns: signal, package, multiplier, max_abs_drift, delta_h, failed.
        ``max_abs_drift`` is ``max_q |h(q, m) - h(q, reference)|``.
    """
    rows = []
    for entry in signals:
        name, x = entry[0], entry[1]
        x = np.asarray(x, dtype=float)
        for pkg, fn in adapters.items():
            base, base_err = _hq(fn, x * reference, scales, q)
            for m in multipliers:
                hq, err = _hq(fn, x * m, scales, q)
                drift = np.abs(hq - base)
                drift = drift[np.isfinite(drift)]
                finite = hq[np.isfinite(hq)]
                rows.append({
                    "signal": name,
                    "package": pkg,
                    "multiplier": float(m),
                    "max_abs_drift": float(drift.max()) if drift.size else np.nan,
                    "delta_h": float(finite.max() - finite.min()) if finite.size else np.nan,
                    "failed": bool(err) or not np.isfinite(hq).any(),
                })
            if verbose:
                sub = [r for r in rows if r["signal"] == name and r["package"] == pkg]
                worst = np.nanmax([r["max_abs_drift"] for r in sub]) if sub else np.nan
                nfail = sum(r["failed"] for r in sub)
                print(f"  {name:<28} {pkg:<18} worst drift={worst:8.4f}  failures={nfail}")
    return pd.DataFrame(rows)

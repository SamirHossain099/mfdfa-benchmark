"""Uniform adapters over the DFA/MFDFA implementations under comparison.

Every adapter exposes the same call signature::

    estimate(x, scales, q) -> dict with keys "hq" (array aligned to q) and "H"

so the driver can treat the packages interchangeably. Where a package returns
fluctuation functions rather than exponents, the adapter fits the log-log slopes
itself using the shared :func:`loglog_slope`, so that differences in reported
exponents come from the package's fluctuation computation rather than from
inconsistent curve fitting on our side.

``fathon`` is handled out of process: its published wheels are compiled against
the NumPy 1.x ABI and fail to import under NumPy 2, so it runs in a separate
interpreter (see ``fathon_worker.py`` and the README).
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np

__all__ = ["ADAPTERS", "loglog_slope", "available_adapters", "package_versions"]


def loglog_slope(scales, values) -> float:
    """Least-squares slope of ``log(values)`` on ``log(scales)``.

    Non-finite and non-positive values are dropped, which is how a user would
    have to handle them anyway. Returns NaN if fewer than two points survive.
    """
    s = np.asarray(scales, dtype=float)
    v = np.asarray(values, dtype=float)
    good = np.isfinite(s) & np.isfinite(v) & (v > 0)
    if good.sum() < 2:
        return float("nan")
    return float(np.polyfit(np.log(s[good]), np.log(v[good]), 1)[0])


def _width(hq) -> float:
    h = np.asarray(hq, dtype=float)
    h = h[np.isfinite(h)]
    return float(h.max() - h.min()) if h.size else float("nan")


# --------------------------------------------------------------------------
# Adapters
# --------------------------------------------------------------------------

def estimate_mfdfa_pkg(x, scales, q):
    """`MFDFA` (Rydin Gorjao et al.). Returns F(q, scale); we fit the slopes."""
    from MFDFA import MFDFA

    lag, fluct = MFDFA(np.asarray(x, float), lag=np.asarray(scales, int),
                       q=np.asarray(q, float), order=1)
    hq = np.array([loglog_slope(lag, fluct[:, j]) for j in range(fluct.shape[1])])
    return {"hq": hq}


def estimate_neurokit2(x, scales, q):
    """`neurokit2.fractal_dfa` in multifractal mode; reports h(q) directly."""
    import neurokit2 as nk

    _, info = nk.fractal_dfa(
        np.asarray(x, float), scale=np.asarray(scales, int),
        q=np.asarray(q, float), multifractal=True, show=False,
    )
    hq = np.asarray(info["h"], dtype=float).ravel()
    return {"hq": hq}


def estimate_fdnkit(x, scales, q, rel_floor=1e-3):
    """`fdnkit.mfdfa`. ``rel_floor=0`` reproduces the unguarded behaviour."""
    from fdnkit.mfdfa import mfdfa

    res = mfdfa(np.asarray(x, float), scales=np.asarray(scales, int),
                q=np.asarray(q, float), order=1, rel_floor=rel_floor,
                check_flat=False)
    return {"hq": np.asarray(res.hq, dtype=float)}


def estimate_fdnkit_unguarded(x, scales, q):
    return estimate_fdnkit(x, scales, q, rel_floor=0.0)


def estimate_nolds(x, scales, q):
    """`nolds.dfa`: monofractal only, so h(q) is undefined and only H is reported."""
    import nolds

    H = float(nolds.dfa(np.asarray(x, float), nvals=np.asarray(scales, int)))
    return {"hq": np.full(len(q), np.nan), "H": H}


def estimate_fathon(x, scales, q, python_exe=None):
    """`fathon`, run out of process against a NumPy 1.x interpreter."""
    exe = python_exe or _fathon_python()
    if exe is None:
        raise RuntimeError(
            "no fathon interpreter configured; set FATHON_PYTHON or pass python_exe"
        )
    worker = str(Path(__file__).with_name("fathon_worker.py"))
    with tempfile.TemporaryDirectory() as tmp:
        sig = Path(tmp) / "signal.npy"
        np.save(sig, np.asarray(x, float))
        cmd = [exe, worker, str(sig),
               ",".join(str(int(s)) for s in scales),
               ",".join(str(float(v)) for v in q)]
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    if out.returncode != 0:
        raise RuntimeError(f"fathon worker failed: {out.stderr.strip()[:400]}")
    payload = json.loads(out.stdout)
    return {"hq": np.asarray(payload["hq"], dtype=float)}


def _fathon_python():
    import os

    exe = os.environ.get("FATHON_PYTHON")
    if exe and Path(exe).exists():
        return exe
    for candidate in (
        Path(tempfile.gettempdir()) / "fathonenv" / "Scripts" / "python.exe",
        Path(tempfile.gettempdir()) / "fathonenv" / "bin" / "python",
    ):
        if candidate.exists():
            return str(candidate)
    return None


ADAPTERS = {
    "MFDFA": estimate_mfdfa_pkg,
    "neurokit2": estimate_neurokit2,
    "fdnkit": estimate_fdnkit,
    "fdnkit_unguarded": estimate_fdnkit_unguarded,
    "nolds": estimate_nolds,
    "fathon": estimate_fathon,
}


def available_adapters() -> dict:
    """Return the adapters that can actually run in this environment."""
    ok = {}
    for name, fn in ADAPTERS.items():
        try:
            probe = np.random.default_rng(0).standard_normal(2048)
            fn(probe, np.array([16, 32, 64, 128]), np.array([-2.0, 2.0]))
            ok[name] = fn
        except Exception:
            continue
    return ok


def package_versions() -> dict:
    """Record installed versions so results are attributable to a environment."""
    import platform

    vers = {"python": platform.python_version()}
    for mod in ("numpy", "scipy", "MFDFA", "neurokit2", "nolds", "fdnkit"):
        try:
            m = __import__(mod)
            vers[mod] = getattr(m, "__version__", "unknown")
        except Exception:
            vers[mod] = "not installed"
    exe = _fathon_python()
    vers["fathon_python"] = exe or "not configured"
    return vers

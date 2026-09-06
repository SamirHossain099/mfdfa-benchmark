"""Driver: run every available implementation over every signal, tidily.

Writes two CSVs into ``results/``:

* ``hq_long.csv``   one row per (signal, package, q) with the estimated h(q)
                    and, where known, the analytic truth;
* ``summary.csv``   one row per (signal, package) with delta_h, h(q=2), and
                    the max error against truth when truth exists.

Usage::

    python -m benchmark.run_benchmark --subjects 1 2 3 --channels 8
    python -m benchmark.run_benchmark --synthetic-only
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .adapters import ADAPTERS, available_adapters, package_versions
from .invariance import MULTIPLIERS, invariance_table
from .signals import load_eegbci_channels, synthetic_cases

RESULTS = Path(__file__).resolve().parents[1] / "results"

DEFAULT_SCALES = np.array([16, 24, 32, 48, 64, 96, 128, 192, 256])
DEFAULT_Q = np.array([-5.0, -3.0, -2.0, -1.0, 1.0, 2.0, 3.0, 5.0])


def _width(hq) -> float:
    h = np.asarray(hq, dtype=float)
    h = h[np.isfinite(h)]
    return float(h.max() - h.min()) if h.size else float("nan")


def run(signals, adapters, scales, q, verbose=True):
    """Return (long_df, summary_df) for the given signals and adapters."""
    long_rows, summary_rows = [], []
    for name, x, *rest in signals:
        truth_fn = rest[0] if rest else None
        truth = np.asarray(truth_fn(q), dtype=float) if truth_fn else None
        for pkg, fn in adapters.items():
            t0 = time.perf_counter()
            try:
                res = fn(x, scales, q)
                hq = np.asarray(res.get("hq"), dtype=float).ravel()
                if hq.size != q.size:  # nolds and any partial reporters
                    hq = np.full(q.size, np.nan) if hq.size != 1 else np.full(q.size, hq[0])
                err = None
            except Exception as exc:  # keep going; record the failure
                hq = np.full(q.size, np.nan)
                err = f"{type(exc).__name__}: {exc}"[:200]
            elapsed = time.perf_counter() - t0

            for j, qq in enumerate(q):
                long_rows.append({
                    "signal": name, "package": pkg, "q": float(qq),
                    "hq": float(hq[j]),
                    "hq_truth": float(truth[j]) if truth is not None else np.nan,
                })
            max_err = np.nan
            if truth is not None:
                d = np.abs(hq - truth)
                d = d[np.isfinite(d)]
                max_err = float(d.max()) if d.size else np.nan
            summary_rows.append({
                "signal": name, "package": pkg,
                "delta_h": _width(hq),
                "h_q2": float(hq[np.argmin(np.abs(q - 2.0))]),
                "max_abs_err_vs_truth": max_err,
                "seconds": round(elapsed, 3),
                "error": err,
            })
            if verbose:
                flag = " ERROR" if err else ""
                print(f"  {name:<34} {pkg:<18} delta_h={_width(hq):8.3f}{flag}")
    return pd.DataFrame(long_rows), pd.DataFrame(summary_rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--subjects", type=int, nargs="*", default=[1],
                    help="EEGBCI subject numbers")
    ap.add_argument("--run", type=int, default=4, help="EEGBCI run number")
    ap.add_argument("--channels", type=int, default=8, help="channels per subject")
    ap.add_argument("--synthetic-only", action="store_true")
    ap.add_argument("--real-only", action="store_true")
    ap.add_argument("--n", type=int, default=20000, help="synthetic signal length")
    ap.add_argument("--packages", nargs="*", default=None,
                    help=f"subset of: {' '.join(ADAPTERS)}")
    ap.add_argument("--out-prefix", default="")
    ap.add_argument("--invariance", action="store_true",
                    help="also test whether h(q) is invariant to rescaling the signal")
    args = ap.parse_args(argv)

    print("Probing available implementations ...")
    adapters = available_adapters()
    if args.packages:
        adapters = {k: v for k, v in adapters.items() if k in set(args.packages)}
    print("  usable:", ", ".join(adapters) or "(none)")
    missing = [k for k in ADAPTERS if k not in adapters]
    if missing:
        print("  unavailable:", ", ".join(missing))
    if not adapters:
        raise SystemExit("no implementations available")

    signals = []
    if not args.real_only:
        signals += synthetic_cases(n=args.n)
    if not args.synthetic_only:
        for s in args.subjects:
            signals += [(n, x) for n, x in
                        load_eegbci_channels(s, args.run, args.channels)]
    print(f"Running {len(signals)} signals x {len(adapters)} implementations ...")

    long_df, summary_df = run(signals, adapters, DEFAULT_SCALES, DEFAULT_Q)

    RESULTS.mkdir(parents=True, exist_ok=True)
    p = args.out_prefix
    long_df.to_csv(RESULTS / f"{p}hq_long.csv", index=False)
    summary_df.to_csv(RESULTS / f"{p}summary.csv", index=False)
    (RESULTS / f"{p}environment.json").write_text(
        json.dumps({"versions": package_versions(),
                    "scales": DEFAULT_SCALES.tolist(),
                    "q": DEFAULT_Q.tolist()}, indent=2), encoding="utf-8")

    print(f"\nWrote {len(long_df)} rows to {RESULTS / (p + 'hq_long.csv')}")

    # Headline: disagreement across implementations, per signal.
    # Report the ABSOLUTE range, not a ratio: when every implementation returns a
    # near-zero delta_h (as on clean monofractal signals) a max/min ratio is
    # dominated by numerical noise and badly overstates the disagreement.
    mono = {"nolds"}  # monofractal-only packages have no delta_h to compare
    piv = summary_df[~summary_df.package.isin(mono)].pivot_table(
        index="signal", columns="package", values="delta_h")
    piv["range"] = piv.max(axis=1) - piv.min(axis=1)
    print("\ndelta_h by implementation (range = max - min across packages):")
    print(piv.round(3).to_string())

    if args.invariance:
        print("\nScale-invariance test: h(q) must not change when the signal is")
        print(f"multiplied by a constant. Multipliers: {', '.join(f'{m:g}' for m in MULTIPLIERS)}")
        inv = invariance_table(signals, adapters, DEFAULT_SCALES, DEFAULT_Q, verbose=False)
        inv.to_csv(RESULTS / f"{p}invariance.csv", index=False)
        print("\nworst |h(q) drift| under rescaling:")
        print(inv.pivot_table(index="signal", columns="package",
                              values="max_abs_drift", aggfunc="max").round(4).to_string())
        print(f"\nmultipliers that failed outright (of {len(MULTIPLIERS)}):")
        print(inv.pivot_table(index="signal", columns="package",
                              values="failed", aggfunc="sum").astype(int).to_string())

    truth = summary_df[summary_df.max_abs_err_vs_truth.notna()]
    truth = truth[~truth.package.isin(mono)]
    if not truth.empty:
        print("\nmax |h(q) - analytic truth| where the answer is known:")
        print(truth.pivot_table(index="signal", columns="package",
                                values="max_abs_err_vs_truth").round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

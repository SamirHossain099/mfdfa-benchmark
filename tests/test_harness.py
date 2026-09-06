"""Tests for the benchmark harness itself.

A reproducibility study is only as trustworthy as its measuring instrument, so
the adapters, the ground-truth formulas, and the invariance machinery are
checked directly. These tests use synthetic signals only and do not touch the
network, so they run anywhere.
"""

from __future__ import annotations

import numpy as np
import pytest

from benchmark.adapters import available_adapters, loglog_slope
from benchmark.invariance import invariance_table
from benchmark.prevalence import prevalence_table
from benchmark.signals import cascade_analytic_hq, synthetic_cases

SCALES = np.array([16, 24, 32, 48, 64, 96, 128])
Q = np.array([-3.0, -1.0, 1.0, 3.0])


# ---------------------------------------------------------------- utilities

def test_loglog_slope_recovers_a_known_power_law():
    s = np.array([2.0, 4, 8, 16, 32, 64])
    assert abs(loglog_slope(s, s**0.75) - 0.75) < 1e-10


def test_loglog_slope_ignores_nonpositive_and_nonfinite():
    s = np.array([2.0, 4, 8, 16])
    v = np.array([2.0**0.5, np.nan, 8**0.5, -1.0])
    # Only two usable points remain, and they still define the 0.5 slope.
    assert abs(loglog_slope(s, v) - 0.5) < 1e-10


def test_loglog_slope_nan_when_underdetermined():
    assert np.isnan(loglog_slope(np.array([4.0]), np.array([2.0])))


# ------------------------------------------------------------- ground truth

def test_cascade_analytic_hq_is_decreasing_in_q():
    q = np.array([-5.0, -2.0, -1.0, 1.0, 2.0, 5.0])
    h = cascade_analytic_hq(q, a=0.3)
    assert np.all(np.isfinite(h))
    assert np.all(np.diff(h) < 0), "h(q) must decrease with q for a multifractal"


def test_cascade_is_monofractal_when_p_is_half():
    # With a = 0.5 the cascade is uniform, so h(q) collapses to a constant.
    h = cascade_analytic_hq(np.array([-5.0, -1.0, 1.0, 5.0]), a=0.5)
    assert np.ptp(h) < 1e-12


def test_cascade_analytic_hq_returns_nan_at_zero():
    assert np.isnan(cascade_analytic_hq(np.array([0.0]))[0])


# ----------------------------------------------------------------- signals

def test_synthetic_cases_have_truth_and_expected_lengths():
    cases = synthetic_cases(n=4096, seed=0)
    names = [c[0] for c in cases]
    assert any(n.startswith("fgn_H0.7_flat") for n in names)
    assert any(n.startswith("cascade") for n in names)
    for name, x, truth in cases:
        assert x.ndim == 1 and x.size >= 4096, name
        assert truth is not None, name
        assert np.all(np.isfinite(truth(Q))) or name.startswith("cascade")


def test_flat_run_cases_actually_contain_flat_runs():
    from fdnkit.preprocessing import find_flat_runs

    cases = {c[0]: c[1] for c in synthetic_cases(n=8192, seed=0)}
    assert find_flat_runs(cases["fgn_H0.7_flat1x32"], min_length=16).shape[0] >= 1
    assert find_flat_runs(cases["fgn_H0.7_clean"] if "fgn_H0.7_clean" in cases
                          else cases["fgn_H0.7"], min_length=16).shape[0] == 0


# ---------------------------------------------------------------- adapters

def test_at_least_one_adapter_is_available():
    assert available_adapters(), "no MFDFA implementation importable"


def test_adapters_agree_on_a_clean_monofractal_signal():
    """The premise of the whole study: on clean ground truth they should agree."""
    from fdnkit.synthetic import fgn

    x = fgn(16384, 0.7, seed=1)
    ad = available_adapters()
    ad.pop("nolds", None)  # monofractal only, no h(q)
    widths = {}
    for pkg, fn in ad.items():
        hq = np.asarray(fn(x, SCALES, Q)["hq"], dtype=float)
        hq = hq[np.isfinite(hq)]
        if hq.size:
            widths[pkg] = hq.max() - hq.min()
    assert len(widths) >= 2
    spread = max(widths.values()) - min(widths.values())
    assert spread < 0.15, f"implementations disagree on a clean signal: {widths}"


# -------------------------------------------------------------- invariance

def test_invariance_table_shape_and_columns():
    from fdnkit.synthetic import fgn

    ad = {k: v for k, v in available_adapters().items() if k == "fdnkit"}
    if not ad:
        pytest.skip("fdnkit adapter unavailable")
    df = invariance_table([("s", fgn(4096, 0.7, seed=0))], ad, SCALES, Q,
                          multipliers=(1.0, 1e-3), verbose=False)
    assert set(["signal", "package", "multiplier", "max_abs_drift", "failed"]) <= set(df.columns)
    assert len(df) == 2


def test_relative_floor_is_scale_invariant_but_absolute_is_not():
    """The paper's central mechanistic claim, as a test."""
    from fdnkit.mfdfa import mfdfa

    from benchmark.signals import synthetic_cases

    x = dict((c[0], c[1]) for c in synthetic_cases(n=16384, seed=1))["fgn_H0.7_flat5x32"]
    rel = [mfdfa(x * m, scales=SCALES, q=Q, rel_floor=1e-3, check_flat=False).delta_h
           for m in (1.0, 1e-5)]
    absol = [mfdfa(x * m, scales=SCALES, q=Q, rel_floor=0.0, check_flat=False).delta_h
             for m in (1.0, 1e-5)]
    assert abs(rel[0] - rel[1]) < 1e-6, "relative floor should be scale invariant"
    assert abs(absol[0] - absol[1]) > abs(rel[0] - rel[1]), \
        "absolute (machine-epsilon) floor should drift more than the relative one"


# -------------------------------------------------------------- prevalence

def test_prevalence_table_records_flat_content():
    from fdnkit.synthetic import fgn

    ad = {k: v for k, v in available_adapters().items() if k == "fdnkit"}
    if not ad:
        pytest.skip("fdnkit adapter unavailable")
    clean = fgn(8192, 0.7, seed=3)
    flat = clean.copy()
    flat[1000:1064] = flat[1000]
    df = prevalence_table([("clean", clean), ("flat", flat)], ad, SCALES, Q, verbose=False)
    assert df[df.signal == "clean"].flat_runs.iloc[0] == 0
    assert df[df.signal == "flat"].flat_runs.iloc[0] >= 1
    assert df[df.signal == "flat"].flat_fraction.iloc[0] > 0

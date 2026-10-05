"""Every number in the manuscript, recomputed from ``results/`` and looked up in the text.

Nothing here hardcodes the number it guards: each value is computed from a results table, formatted
the way the manuscript prints it, and asserted to appear in ``MANUSCRIPT.md``. If a result moves,
the formatted string moves with it and the test names the sentence that is now wrong.

The manuscript is not part of this repository, so the module is skipped when it is absent; the
computations still document where each reported number comes from.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
MANUSCRIPT = ROOT / "MANUSCRIPT.md"

pytestmark = pytest.mark.skipif(not MANUSCRIPT.exists(), reason="manuscript not in this checkout")

PKGS = ["MFDFA", "fathon", "fdnkit", "fdnkit_unguarded", "neurokit2"]
SUP = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def ms() -> str:
    return re.sub(r"\s+", " ", MANUSCRIPT.read_text(encoding="utf-8"))


def sci(x: float, digits: int = 1) -> str:
    """1.0018e-12 -> '1.0 × 10⁻¹²', the manuscript's form."""
    e = int(np.floor(np.log10(abs(x))))
    return f"{x / 10**e:.{digits}f} × 10{str(e).translate(SUP)}"


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def present(*fragments: str) -> None:
    text = ms()
    for f in fragments:
        assert f in text, f"manuscript does not contain {f!r}"


def csv(name: str) -> pd.DataFrame:
    return pd.read_csv(RESULTS / name)


def with_subject(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(subject=df.signal.str.extract(r"_s(\d+)_")[0])


# ---------------------------------------------------------------- 3.1 known exponents
CLEAN = {"fgn_H0.3": "fGn H = 0.3", "fgn_H0.5": "fGn H = 0.5", "fgn_H0.7": "fGn H = 0.7",
         "fgn_H0.9": "fGn H = 0.9", "white_noise": "white noise",
         "cascade_p0.3": "binomial cascade"}
FLAT = {"fgn_H0.7_flat1x32": "one run of 32", "fgn_H0.7_flat5x32": "five runs of 32",
        "fgn_H0.7_flat20x64": "twenty runs of 64"}


def table_rows(signals: dict) -> list[str]:
    s = csv("summary.csv").set_index(["signal", "package"]).max_abs_err_vs_truth
    return ["| " + label + " | " + " | ".join(f"{s[(sig, p)]:.3f}" for p in PKGS) + " |"
            for sig, label in signals.items()]


def test_table1_known_exponents():
    present(*table_rows(CLEAN))


def test_worst_error_and_cascade_spread():
    s = csv("summary.csv")
    clean = s[s.signal.isin(CLEAN) & s.package.isin(PKGS)]
    worst = clean.max_abs_err_vs_truth.max()
    casc = s[(s.signal == "cascade_p0.3") & s.package.isin(PKGS)].delta_h
    present(f"off by more than {worst:.3f}", f"the spread between implementations is "
            f"{casc.max() - casc.min():.3f}", f"to within {worst:.3f} of the true value")


def test_fathon_fdnkit_machine_precision():
    hq = csv("hq_long.csv")
    a = hq[(hq.package == "fathon") & hq.signal.isin(CLEAN)].set_index(["signal", "q"]).hq
    b = hq[(hq.package == "fdnkit") & hq.signal.isin(CLEAN)].set_index(["signal", "q"]).hq
    per = (a - b).abs().groupby(level=0).max()
    assert per.idxmax() == "cascade_p0.3"
    assert per.drop("cascade_p0.3").max() < 1e-13
    present(f"largest disagreement across the six controls is {sci(per.max())}, on the cascade")


# ---------------------------------------------------------------- 3.2 recorded EEG
def test_single_channel_widths():
    s = csv("summary.csv")
    v = sorted(s[(s.signal == "eegbci_s1_r4_Fc5") & s.package.isin(PKGS)].delta_h)
    present("widths of " + ", ".join(f"{x:.3f}" for x in v[:-1]) + f" and {v[-1]:.3f}",
            f"The range, {v[-1] - v[0]:.1f},", f"ranged from {v[0]:.2f} to {v[-1]:.2f}")


# ---------------------------------------------------------------- 3.3 constant runs
def test_table2_constant_run_controls():
    present(*table_rows(FLAT))
    s = csv("summary.csv").set_index(["signal", "package"]).max_abs_err_vs_truth
    one = max(s[("fgn_H0.7_flat1x32", p)] for p in PKGS)
    present(f"by up to {one:.3f} against a true value of 0.7", f"by up to {one:.1f} against",
            f"erring by {s[('fgn_H0.7_flat1x32', 'fathon')]:.3f}",
            f"by {s[('fgn_H0.7_flat5x32', 'fathon')]:.3f} with five", f"{32 / 20000:.2%} of the data")


def test_prevalence_counts_and_intervals():
    ch = with_subject(csv("prevalence.csv")).drop_duplicates("signal")
    k_ch, n_ch = int((ch.flat_runs > 0).sum()), len(ch)
    by = ch.groupby("subject").flat_runs.agg(lambda v: (v > 0).mean())
    assert set(by.unique()) <= {0.0, 1.0}  # every subject is affected on all channels or none
    k_s, n_s = int((by == 1).sum()), len(by)
    present(f"{n_ch} channels of the motor-imagery corpus, {k_ch} contain",
            f"exactly {k_s} of the {n_s} subjects", f"at most {ch.flat_fraction.max():.1%} of samples")
    for k, n in ((k_s, n_s), (k_ch, n_ch)):
        ci = stats.binomtest(k, n).proportion_ci(method="exact")
        present(f"{pct(ci.low)} to {pct(ci.high)}")


def test_exceedance_and_subject_spearman():
    p = with_subject(csv("prevalence.csv"))
    flat, non = p[p.flat_runs > 0], p[p.flat_runs == 0]
    exc = {k: (flat[flat.package == k].delta_h > 1).mean() for k in PKGS}
    assert exc["MFDFA"] == exc["fdnkit"] == exc["fdnkit_unguarded"] == 1.0
    assert all((non[non.package == k].delta_h > 1).mean() == 0 for k in PKGS)
    present(f"as do {pct(exc['fathon'])} of `fathon`")

    def subj(pkg):
        s = p[p.package == pkg].groupby("subject").agg(ff=("flat_fraction", "mean"),
                                                       dh=("delta_h", "mean"))
        return stats.spearmanr(s.ff, s.dh)

    guarded = {k: subj(k) for k in ("MFDFA", "fathon", "fdnkit", "fdnkit_unguarded")}
    r = guarded["MFDFA"]
    assert all(abs(g.statistic - r.statistic) < 1e-12 for g in guarded.values())
    nk = subj("neurokit2")
    present(f"width is {r.statistic:.3f} for", f"(p = {r.pvalue:.3f}, n = 10)",
            f"−{abs(nk.statistic):.3f} on subject means", f"(p = {nk.pvalue:.3f})")


# ---------------------------------------------------------------- 3.4 invariance
def test_table3_invariance_and_failures():
    t = csv("invariance_two_corpora.csv")
    t["corpus"] = t.signal.str.split("::").str[0]
    order = PKGS + ["nolds"]
    rows = {"eegbci": "motor imagery, 160 Hz", "sleepedf": "Sleep-EDF, 100 Hz"}
    for corpus, label in rows.items():
        d = t[t.corpus == corpus].groupby("package").max_abs_drift.max()
        present("| " + label + " | " + " | ".join(f"{d[p]:.3f}" for p in order) + " |")
        f = t[t.corpus == corpus].groupby("package").failed.mean()
        assert (f.drop("neurokit2") == 0).all()
    f_s = t[(t.corpus == "sleepedf") & (t.package == "neurokit2")].failed.mean()
    f_e = t[(t.corpus == "eegbci") & (t.package == "neurokit2")].failed.mean()
    present(f"no result for {pct(f_s)} of signal-multiplier", f"figures are {pct(f_e)} and 0.0%",
            f"drifted by {t[t.corpus == 'sleepedf'].max_abs_drift.max():.3f} under rescaling")


def test_white_noise_amplitude_dependence():
    i = csv("invariance.csv")
    w = i[(i.signal == "white_noise") & (i.package == "neurokit2")].set_index("multiplier")
    ok = w.loc[[1.0, 1e-3]].delta_h
    assert ok.max() - ok.min() < 5e-4
    assert bool(w.loc[1e-5, "failed"]) and not bool(w.loc[1e-4, "failed"])
    present(f"width of {ok.iloc[0]:.3f} for multipliers from 1 down to",
            f"{w.loc[1e-4, 'delta_h']:.3f} at 10⁻⁴")


# ---------------------------------------------------------------- 3.5 discard fractions
def test_table4_discard():
    d = csv("discard_summary.csv").set_index("corpus")
    for corpus in ("motor imagery", "Sleep-EDF"):
        r = d.loc[corpus]
        present(f"| {corpus} | {sci(r.median_signal_sd, 2)} | {pct(r.discarded_16)} | "
                f"{pct(r.discarded_64)} | {pct(r.discarded_256)} | {pct(r.failure_rate)} |")
    m, s = d.loc["motor imagery"], d.loc["Sleep-EDF"]
    present(f"falls from {pct(m.discarded_16)} at scale 16 to {pct(m.discarded_256)}",
            f"{pct(s.discarded_16)} down to {pct(s.discarded_256)}")


def test_nan_rates_match_prevalence_tables():
    d = csv("discard_summary.csv").set_index("corpus").failure_rate
    for name, corpus in (("prevalence.csv", "motor imagery"), ("prevalence_sleepedf.csv", "Sleep-EDF")):
        p = csv(name)
        nan = p[p.package == "neurokit2"].delta_h.isna().mean()
        assert abs(nan - d[corpus]) < 1e-9
    present(f"no result for {pct(d['Sleep-EDF'])} of Sleep-EDF epochs against {pct(d['motor imagery'])}")


# ---------------------------------------------------------------- 3.6 second corpus
def test_sleepedf_replication():
    s = with_subject(csv("prevalence_sleepedf.csv"))
    ch = s.drop_duplicates("signal")
    k = int((ch.flat_runs > 0).sum())
    ks = ch[ch.flat_runs > 0].subject.nunique()
    assert s.delta_h.max() < 1
    four = s[s.package.isin(["MFDFA", "fathon", "fdnkit", "fdnkit_unguarded"])]
    spread = four.groupby("signal").delta_h.agg(lambda v: v.max() - v.min())
    m = csv("prevalence.csv")
    r_m = stats.spearmanr(m[m.package == "MFDFA"].flat_fraction, m[m.package == "MFDFA"].delta_h)
    r_s = stats.spearmanr(s[s.package == "MFDFA"].flat_fraction, s[s.package == "MFDFA"].delta_h)
    fisher = stats.fisher_exact([[3, 7], [ks, ch.subject.nunique() - ks]]).pvalue
    present(f"Only {k} of {len(ch)} epochs", f"belongs to {ks} of the {ch.subject.nunique()} subjects",
            f"median of {spread.median():.3f} across epochs and by at most {spread.max():.3f}",
            f"falls from {r_m.statistic:.3f} to {r_s.statistic:.3f}", f"gives p = {fisher:.3f}")


# ---------------------------------------------------------------- 4.2 the relative floor
def test_relative_floor_caps_width():
    p = csv("prevalence.csv")
    flat = p[p.flat_runs > 0].groupby("package").delta_h
    assert flat.min()["fdnkit"] > 1
    present(f"caps the worst width at {flat.max()['fdnkit']:.2f} against {flat.max()['MFDFA']:.2f}")

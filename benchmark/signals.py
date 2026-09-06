"""Test signals: synthetic controls with known answers, and real recordings.

The synthetic cases are the reference condition. If implementations agree on
signals whose exponents are known analytically but disagree on real recordings,
the disagreement is a property of the data rather than of differing but
defensible conventions.

Ground truth available here:

* fractional Gaussian noise is monofractal, so ``h(q) = H`` for every ``q``;
* the binomial multiplicative cascade has a closed-form generalized Hurst
  exponent (Kantelhardt et al. 2002),
  ``h(q) = 1/q - ln(a**q + (1-a)**q) / (q * ln 2)``.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "cascade_analytic_hq",
    "synthetic_cases",
    "load_eegbci_channels",
    "load_sleepedf_epochs",
]


def cascade_analytic_hq(q, a: float = 0.3) -> np.ndarray:
    """Closed-form ``h(q)`` for a binomial multiplicative cascade.

    Parameters
    ----------
    q : array-like
        Moment orders. ``q = 0`` is returned as NaN (the limit needs L'Hopital).
    a : float
        Cascade multiplier in (0, 1); the other branch is ``1 - a``.
    """
    q = np.asarray(q, dtype=float)
    out = np.full(q.shape, np.nan)
    nz = q != 0
    qq = q[nz]
    out[nz] = 1.0 / qq - np.log(a**qq + (1 - a) ** qq) / (qq * np.log(2.0))
    return out


def synthetic_cases(n: int = 20000, seed: int = 1, cascade_levels: int = 14,
                    cascade_p: float = 0.3):
    """Yield ``(name, signal, truth)`` synthetic controls.

    ``truth`` is a callable ``q -> h(q)`` where the answer is known, else None.
    """
    from fdnkit.synthetic import binomial_cascade, fgn

    cases = []
    for H in (0.3, 0.5, 0.7, 0.9):
        cases.append((
            f"fgn_H{H}",
            fgn(n, H, seed=seed),
            (lambda q, _H=H: np.full(np.shape(q), float(_H))),
        ))
    cases.append((
        f"cascade_p{cascade_p}",
        binomial_cascade(cascade_levels, cascade_p, seed=seed),
        (lambda q, _a=cascade_p: cascade_analytic_hq(q, _a)),
    ))
    rng = np.random.default_rng(seed)
    cases.append(("white_noise", rng.standard_normal(n),
                  (lambda q: np.full(np.shape(q), 0.5))))

    # Controls that isolate the suspected trigger: constant runs.
    base = fgn(n, 0.7, seed=seed)
    for n_runs, run_len in ((1, 32), (5, 32), (20, 64)):
        y = base.copy()
        # Spread the runs evenly so the layout is independent of signal length.
        for i in np.linspace(0, n - run_len - 1, n_runs + 2)[1:-1].astype(int):
            y[i:i + run_len] = y[i]
        cases.append((f"fgn_H0.7_flat{n_runs}x{run_len}", y,
                      (lambda q: np.full(np.shape(q), 0.7))))
    return cases


def load_eegbci_channels(subject: int = 1, run: int = 4, n_channels: int = 8,
                         max_samples: int | None = None):
    """Yield ``(name, signal)`` for channels of one PhysioNet EEGBCI run.

    Downloads through MNE on first use and caches thereafter. Public data, no
    registration required.
    """
    from mne.datasets import eegbci

    from fdnkit.io import load_edf

    paths = eegbci.load_data(subjects=subject, runs=[run], update_path=True)
    rec = load_edf(paths[0])
    names = [n.strip(". ") for n in rec.channel_names]
    out = []
    for i in range(min(n_channels, rec.n_channels)):
        x = np.asarray(rec.signals[i], dtype=float)
        if max_samples:
            x = x[:max_samples]
        out.append((f"eegbci_s{subject}_r{run}_{names[i]}", x))
    return out


def load_sleepedf_epochs(subject: int = 0, recording: int = 1, n_epochs: int = 4,
                         epoch_samples: int = 20000, channels=("EEG Fpz-Cz", "EEG Pz-Oz")):
    """Yield ``(name, signal)`` epochs from a PhysioNet Sleep-EDF recording.

    A second corpus recorded on different hardware to EEGBCI: 100 Hz rather than
    160 Hz, two EEG derivations rather than a 64-channel cap, and a
    whole-night rather than a task recording. Using it guards against a finding
    that is specific to one dataset.

    Epochs of ``epoch_samples`` are taken evenly spaced through the night, so a
    single long record contributes several independent segments of a length
    comparable to the EEGBCI channels.

    Public data, fetched and cached through MNE. No registration required.
    """
    import mne
    from mne.datasets.sleep_physionet.age import fetch_data

    paths = fetch_data(subjects=[subject], recording=[recording],
                       on_missing="warn", verbose="ERROR")
    psg = paths[0][0]
    raw = mne.io.read_raw_edf(psg, preload=True, verbose="ERROR")

    out = []
    n = raw.n_times
    for ch in channels:
        if ch not in raw.ch_names:
            continue
        x = raw.get_data(picks=[ch])[0].astype(float)
        # Skip the first and last 10% of the night: lights-on periods often
        # contain long flat or saturated stretches that are not representative.
        lo, hi = int(0.1 * n), int(0.9 * n) - epoch_samples
        if hi <= lo:
            continue
        for k, start in enumerate(np.linspace(lo, hi, n_epochs).astype(int)):
            seg = x[start:start + epoch_samples]
            if seg.size == epoch_samples:
                tag = ch.replace("EEG ", "").replace(" ", "")
                out.append((f"sleepedf_s{subject}_r{recording}_{tag}_e{k}", seg))
    return out

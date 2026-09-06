# Cross-implementation benchmark of fractal and multifractal EEG measures

Do the DFA and MFDFA numbers in the EEG literature depend on which software
package produced them?

This repository runs five Python implementations over the same signals with the
same parameters, and compares the generalized Hurst exponents they report. It
uses only public data and synthetic signals with analytically known answers.

## Headline result so far

Signals whose answer is known analytically produce close agreement. Real EEG
does not.

| signal | MFDFA | fathon | fdnkit | fdnkit unguarded | neurokit2 | range |
|---|---|---|---|---|---|---|
| fgn H=0.3 (monofractal) | 0.041 | 0.045 | 0.045 | 0.045 | 0.040 | **0.005** |
| fgn H=0.7 (monofractal) | 0.037 | 0.047 | 0.047 | 0.047 | 0.021 | **0.026** |
| white noise | 0.001 | 0.004 | 0.004 | 0.004 | 0.008 | **0.007** |
| binomial cascade (multifractal) | 0.707 | 0.807 | 0.807 | 0.807 | 0.753 | **0.100** |
| fgn H=0.7 + one 32-sample flat run | 9.896 | 8.615 | 1.801 | 9.771 | 0.024 | **9.872** |
| fgn H=0.7 + five 32-sample flat runs | 11.070 | 0.017 | 2.283 | 10.623 | 1.602 | **11.054** |
| real EEG (EEGBCI s1 Fc5) | 18.388 | 14.644 | 2.495 | 11.598 | 0.479 | **17.909** |
| real EEG (EEGBCI s1 Fc3) | 17.180 | 13.595 | 2.571 | 11.723 | 0.392 | **16.788** |

Values are the multifractal width `delta_h = max h(q) - min h(q)` over
`q` in {-5, -3, -2, -1, 1, 2, 3, 5}, scales 16 to 256, linear detrending.
`range` is the absolute spread across implementations.

Measured against analytic truth, where it exists:

| signal | MFDFA | fathon | fdnkit | fdnkit unguarded | neurokit2 |
|---|---|---|---|---|---|
| binomial cascade | 0.125 | 0.054 | 0.054 | 0.054 | 0.055 |
| fgn H=0.7 | 0.020 | 0.030 | 0.030 | 0.030 | 0.019 |
| fgn H=0.7 + one flat run | **9.880** | **8.590** | 1.777 | **9.747** | 0.016 |

Reading of these numbers:

1. **On clean signals the implementations agree**, and all land within about
   0.13 of the analytic answer. The comparison is therefore fair: differences
   later are not artifacts of mismatched conventions or grids.
2. **A single 32-sample constant run in a 20,000-sample series**, 0.16% of the
   data, moves most implementations by up to 9.9 in `h(q)` against a known
   truth of 0.7. `neurokit2` is essentially unaffected (0.016).
3. **The failure modes differ.** Most implementations inflate `delta_h`.
   `fathon` is bimodal: 8.6 on one flat run, but 0.017 on five, meaning it
   sometimes collapses instead of inflating.
4. **On real EEG the spread reaches 17.9**, which is larger than any plausible
   multifractal width for the signal. A reader of the literature cannot tell
   which tool produced a published number.

These are pilot results on one subject. Prevalence across subjects and corpora
is the obvious next step.

## Second result: scale invariance

`h(q)` is defined through how fluctuations scale with window size, so
multiplying a signal by a positive constant must leave it unchanged. Expressing
an EEG channel in volts rather than microvolts is exactly such a rescaling.

Worst `|h(q)| drift` as the input is rescaled over `1e-6` to `1e+6`:

| signal | MFDFA | fathon | fdnkit | fdnkit unguarded | neurokit2 | nolds |
|---|---|---|---|---|---|---|
| clean fgn H=0.7 | 0.000 | 0.000 | 0.000 | 0.000 | **0.151** | 0.000 |
| white noise | 0.000 | 0.000 | 0.000 | 0.000 | **0.145** | 0.000 |
| real EEG (Fc5) | 0.268 | 0.113 | **0.001** | **6.612** | **0.663** | 0.000 |

Multipliers that failed outright, of seven:

| signal | neurokit2 | all others |
|---|---|---|
| clean fgn H=0.7 | 2 | 0 |
| white noise | 2 | 0 |
| real EEG (Fc5) | 4 | 0 |

Three things follow.

1. **On clean signals only `neurokit2` breaks invariance**, and it also fails
   outright at small amplitudes. Its `_fractal_dfa_fluctuation` discards
   segments using an absolute threshold, `var = var[var > 1e-08]`. The guard
   itself is sound and is more than most implementations do; the defect is that
   the threshold is absolute, so it depends on the units the data is stored in.
   On volt-scale EEG from MNE it discarded 96.6% of segments at scale 16 and 0%
   at scale 256, which biases the log-log slope directly.
2. **`fdnkit` with `rel_floor=0` is the worst offender on real data**, drifting
   by 6.6. That is this project's own code, and the reason is the same: with the
   relative floor disabled it falls back to a machine-epsilon *absolute* floor.
   The defect is the absolute threshold, not any particular package.
3. **A scale-relative floor fixes it.** `fdnkit` with its default
   `rel_floor=1e-3` drifts by 0.0007 on the same data, while still suppressing
   the flat-run divergence.

The general lesson is that any absolute threshold on fluctuation magnitude
breaks a defining invariance of the measure. The threshold has to be relative to
the data at each scale.

## Third result: prevalence across a cohort

80 channels from 10 EEGBCI subjects, same grid.

| implementation | median delta_h | max delta_h | % of channels over 1.0 | % NaN |
|---|---|---|---|---|
| MFDFA | 0.241 | 20.041 | 30.0 | 0.0 |
| fathon | 0.236 | 16.518 | 28.8 | 0.0 |
| fdnkit | 0.241 | 2.691 | 30.0 | 0.0 |
| fdnkit unguarded | 0.241 | 11.888 | 30.0 | 0.0 |
| neurokit2 | 0.426 | 0.777 | 0.0 | 7.5 |

**24 of 80 channels (30.0%) contain a constant run of at least 16 samples.** The
median flat fraction is 0, the maximum only 0.004, so where it happens it
involves at most 0.4% of samples.

Splitting channels by whether they contain any flat run separates the outcomes
almost perfectly:

| implementation | delta_h > 1 among channels WITH flat runs | among channels WITHOUT |
|---|---|---|
| MFDFA | 100.0% | 0.0% |
| fathon | 95.8% | 0.0% |
| fdnkit | 100.0% | 0.0% |
| fdnkit unguarded | 100.0% | 0.0% |
| neurokit2 | 0.0% | 0.0% |

Spearman correlation between a channel's flat fraction and its `delta_h`:
0.794 for MFDFA, fdnkit and fdnkit unguarded, 0.728 for fathon, and **-0.526**
for neurokit2.

Points to draw from this:

1. **Constant runs are common, not exotic.** They occur in 30% of channels of a
   standard public dataset, and their presence predicts implausible output
   essentially deterministically.
2. **neurokit2 fails in the opposite direction.** Its correlation is negative:
   the more flat content a channel has, the more segments its absolute threshold
   discards, pushing `delta_h` down rather than up. It never exceeds 1.0, but it
   also returned NaN for 7.5% of channels.
3. **A scale-relative floor reduces the damage without eliminating it.**
   `fdnkit` caps the maximum at 2.69 rather than 20.04, but 30% of channels
   still exceed 1.0. The floor limits how far a degenerate segment can distort
   the estimate; it does not make the affected channels trustworthy. Screening
   for flat runs and excluding or repairing those channels is still necessary.

## Fourth result: the second corpus splits the two findings apart

80 epochs from 8 Sleep-EDF subjects, recorded at 100 Hz on different hardware to
EEGBCI, same analysis grid.

| implementation | median delta_h | max delta_h | % over 1.0 | % NaN |
|---|---|---|---|---|
| MFDFA | 0.260 | 0.596 | 0.0 | 0.00 |
| fathon | 0.266 | 0.610 | 0.0 | 0.00 |
| fdnkit | 0.266 | 0.610 | 0.0 | 0.00 |
| fdnkit unguarded | 0.266 | 0.610 | 0.0 | 0.00 |
| neurokit2 | 0.435 | 0.648 | 0.0 | **66.25** |

**The flat-run finding does not replicate.** Only 1 of 80 Sleep-EDF epochs
(1.2%) contains a constant run, against 30% of EEGBCI channels. With almost
nothing to trigger it, no implementation exceeds `delta_h` of 1.0, all four
agree to within 0.014, and `fdnkit` with and without its floor return identical
numbers. The Spearman association falls from 0.79 to 0.14, which is what a
near-constant predictor should do.

So the earlier "30% of channels" figure is a property of **EEGBCI, not of EEG**.
Constant runs are a real and consequential defect where they occur, but their
prevalence is dataset specific and cannot be generalised from one corpus. This
is a limitation of the flat-run result, and it is the reason a second corpus was
necessary.

**The amplitude finding replicates, and gets worse.** neurokit2 returned NaN for
**66.25%** of Sleep-EDF epochs, against 7.5% of EEGBCI channels. The mechanism
accounts for it exactly: Sleep-EDF EEG has a smaller median amplitude
(std 2.09e-05 against 5.61e-05), so more segments fall under the absolute
`var > 1e-08` threshold.

Fraction of segments discarded:

| corpus | scale 16 | scale 64 | scale 256 |
|---|---|---|---|
| EEGBCI | 96.6% | 15.1% | 0.0% |
| Sleep-EDF | 99.8% | 39.1% | 0.0% |

The lower the signal amplitude, the more of the recording is silently thrown
away, until the function cannot return an answer at all.

**What this does to the paper.** The amplitude dependence is the primary result:
it is a violated invariance, it reproduces in six lines with no data at all, and
it worsens predictably on a second corpus. The constant-run divergence is a
secondary result: real, mechanistically clear, but of dataset-dependent
prevalence. Presenting them the other way round would overstate what the
evidence supports.

### Invariance on the second corpus isolates the defect cleanly

Repeating the rescaling test on Sleep-EDF removes the constant-run confound,
because that corpus has almost none. Worst `|h(q)| drift` over multipliers
`1e-6` to `1e+6`:

| corpus | MFDFA | fathon | fdnkit | fdnkit unguarded | neurokit2 | nolds |
|---|---|---|---|---|---|---|
| EEGBCI | 2.543 | 0.250 | 0.001 | 6.612 | 0.686 | 0.000 |
| Sleep-EDF | **0.000** | **0.000** | **0.000** | **0.000** | **1.127** | **0.000** |

Percentage of (signal, multiplier) combinations that returned nothing at all:

| corpus | neurokit2 | every other implementation |
|---|---|---|
| EEGBCI | 57.1% | 0.0% |
| Sleep-EDF | 64.3% | 0.0% |

On EEGBCI the picture is muddied: constant runs interact with the absolute
floors, so MFDFA and unguarded fdnkit also drift. On Sleep-EDF, with that
confound absent, **every implementation is exactly invariant except neurokit2**,
which drifts by 1.13 and fails on roughly two thirds of attempts.

This is the cleanest statement of the primary result. It is not a general
fragility of MFDFA under rescaling; it is one implementation, and the others
demonstrate that getting this right is entirely achievable.

## Layout

```
benchmark/
  adapters.py         uniform estimate(x, scales, q) over each package
  fathon_worker.py    out-of-process worker for the NumPy 1.x environment
  signals.py          synthetic controls with analytic truth, plus EEG loaders
  invariance.py       does h(q) survive rescaling the input?
  run_benchmark.py    driver, writes tidy CSVs to results/
  prevalence.py       cohort run; relates delta_h to flat-run content
  figures.py          builds the PNGs in figures/ from the CSVs
results/              hq_long.csv, summary.csv, invariance.csv, prevalence.csv
figures/              fig1 agreement, fig2 h(q) curves, fig3 invariance,
                      fig4 prevalence, fig5 cross-corpus
tests/                harness tests, synthetic only, no network
```

`results/environment.json` records package versions and the parameter grid, so
any result can be attributed to a specific environment.

## Install

```bash
python -m venv .venv && .venv/Scripts/activate     # Windows
pip install -r requirements.txt
```

`fathon` needs its own environment, because its published wheels are compiled
against the NumPy 1.x ABI and fail to import under NumPy 2:

```bash
python -m venv .venv-fathon
.venv-fathon/Scripts/pip install -r requirements-fathon.txt
set FATHON_PYTHON=<abs path to .venv-fathon/Scripts/python.exe>
```

The driver skips any implementation it cannot run, so `fathon` is optional.

## Run

```bash
python -m benchmark.run_benchmark --synthetic-only
python -m benchmark.run_benchmark --subjects 1 2 3 --channels 8
python -m benchmark.run_benchmark --packages MFDFA neurokit2 fdnkit
python -m benchmark.run_benchmark --real-only --invariance   # scale-invariance test
```

EEG is fetched through MNE from PhysioNet on first use and cached. No
registration or data agreement is required.

## Tests

```bash
python -m pytest tests/ -q
```

13 tests covering the harness itself, since a reproducibility study is only as
trustworthy as its measuring instrument. They use synthetic signals only and do
not touch the network. Two of them encode the study's central claims directly:
`test_adapters_agree_on_a_clean_monofractal_signal` asserts the implementations
land within 0.15 of each other on ground truth, which is the premise that makes
the later divergence meaningful; and
`test_relative_floor_is_scale_invariant_but_absolute_is_not` asserts that a
relative floor survives rescaling while a machine-epsilon absolute floor does
not.

## Method notes

* Where a package returns fluctuation functions rather than exponents
  (`MFDFA`), the adapter fits log-log slopes with the shared `loglog_slope`, so
  differences reflect the fluctuation computation rather than our curve fitting.
* `nolds` is monofractal only. It contributes an `H` comparison, is excluded
  from `delta_h` columns, and in the invariance test is measured on the `H` it
  does report rather than being marked as failing for a capability it never
  claimed.
* Disagreement is reported as an **absolute range**, not a ratio. On clean
  signals every implementation returns a near-zero `delta_h`, and a max/min
  ratio there is dominated by numerical noise and badly overstates the spread.
* `fathon` requires `int64` window arrays; NumPy's default integer is `int32` on
  Windows, which raises a buffer dtype error. The worker casts explicitly.

## Reported upstream

Both defects were reported to their maintainers before any write-up:

* `MFDFA`: [LRydin/MFDFA#38](https://github.com/LRydin/MFDFA/issues/38),
  constant runs diverge under negative `q` with no guard and no warning.
* `neurokit2`: [neuropsychology/NeuroKit#1208](https://github.com/neuropsychology/NeuroKit/issues/1208),
  the guard exists but its threshold is absolute, so results depend on input
  units and fail outright on low-amplitude data.

Both reports include a self-contained reproducer and suggest fixes in
increasing order of intrusiveness.

## Background

The zero-variance mechanism behind the negative-`q` divergence is documented:
see Ludescher et al. (2011), *Physica A* 390:2480, on spurious and corrupted
multifractality, and arXiv:2603.04609 for the explicit zero-local-variance
argument. What this benchmark adds is the practical consequence: how far apart
real, widely used implementations land on real recordings, and which
implementation choices drive the difference.

## Status

Pilot results across 10 subjects (80 channels) and five implementations, with
synthetic controls carrying analytic ground truth. Enough to establish the
effect, its mechanism, and its prevalence in one corpus.

The second corpus is done, and it separated the two findings: the amplitude
dependence replicated and intensified, the constant-run prevalence did not.

Both defects have been reported upstream, linked above. Open questions: whether
a third corpus is worth adding to bound how much flat-run prevalence varies
between datasets, and whether the affected channels change any downstream
conclusion.

A write-up of these results is in preparation. It is kept outside this
repository, which holds the code, the derived tables and the figures, so that
the analysis can be rerun independently of the text.

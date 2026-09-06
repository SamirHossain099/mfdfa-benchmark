"""Regenerate Table 4: the fraction of segments `neurokit2`'s absolute threshold removes.

Writes `results/discard.csv` (one row per signal) and `results/discard_summary.csv` (one row per
corpus, which is the table as printed). Uses the same corpora, the same scale grid and the same
`q` grid as the main benchmark, read from `results/environment.json` so the two cannot drift apart.

    python run_discard.py
"""
from __future__ import annotations

import csv
import json
import os
import warnings

warnings.filterwarnings("ignore")

from benchmark.discard import survey  # noqa: E402
from benchmark.signals import load_eegbci_channels, load_sleepedf_epochs  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
REPORT_SCALES = (16, 64, 256)


def main():
    env = json.load(open(os.path.join(RESULTS, "environment.json"), encoding="utf-8"))
    scales, q = env["scales"], env["q"]
    print(f"scales {scales}")
    print(f"q      {q}\n")

    eeg = []
    for subject in range(1, 11):
        eeg += list(load_eegbci_channels(subject=subject, run=4, n_channels=8, max_samples=20000))

    sleep = []
    for subject in range(8):
        sleep += list(load_sleepedf_epochs(subject=subject, recording=1, n_epochs=5,
                                           epoch_samples=20000,
                                           channels=("EEG Fpz-Cz", "EEG Pz-Oz")))

    all_rows, summaries = [], []
    for pairs, corpus in ((eeg, "motor imagery"), (sleep, "Sleep-EDF")):
        rows, summary = survey(pairs, scales, q, REPORT_SCALES)
        for r in rows:
            r["corpus"] = corpus
        all_rows += rows
        summary["corpus"] = corpus
        summaries.append(summary)
        print(f"{corpus}: n = {summary['n']}")
        print(f"  median signal SD   {summary['median_signal_sd']:.3g}")
        for w in REPORT_SCALES:
            print(f"  scale {w:<4d} discarded {100 * summary[f'discarded_{w}']:.1f}%")
        print(f"  returning no result {100 * summary['failure_rate']:.1f}%\n")

    per = os.path.join(RESULTS, "discard.csv")
    with open(per, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["corpus", "signal", "signal_sd", "failed"]
                           + [f"discarded_{s}" for s in REPORT_SCALES])
        w.writeheader()
        w.writerows(all_rows)

    summ = os.path.join(RESULTS, "discard_summary.csv")
    with open(summ, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["corpus", "n", "median_signal_sd"]
                           + [f"discarded_{s}" for s in REPORT_SCALES] + ["failure_rate"])
        w.writeheader()
        w.writerows(summaries)

    print(f"wrote {per}\nwrote {summ}")
    print("\nTable 4, as measured:")
    print("| corpus | median signal SD | " + " | ".join(f"scale {s}" for s in REPORT_SCALES)
          + " | epochs returning NaN |")
    for s in summaries:
        sd = f"{s['median_signal_sd']:.2e}".replace("e-0", " x 10^-")
        cells = " | ".join(f"{100 * s[f'discarded_{w}']:.1f}%" for w in REPORT_SCALES)
        print(f"| {s['corpus']} | {sd} | {cells} | {100 * s['failure_rate']:.1f}% |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

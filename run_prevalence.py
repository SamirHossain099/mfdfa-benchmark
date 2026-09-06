"""Cohort prevalence run across EEGBCI subjects."""
import warnings; warnings.filterwarnings("ignore")
import argparse
import numpy as np, pandas as pd
from benchmark.adapters import available_adapters
from benchmark.prevalence import prevalence_table, summarize_prevalence
from benchmark.signals import load_eegbci_channels

SC = np.array([16,24,32,48,64,96,128,192,256])
Q  = np.array([-5.,-3.,-2.,-1.,1.,2.,3.,5.])

ap = argparse.ArgumentParser()
ap.add_argument("--subjects", type=int, default=10)
ap.add_argument("--channels", type=int, default=8)
ap.add_argument("--run", type=int, default=4)
a = ap.parse_args()

ad = available_adapters()
ad.pop("nolds", None)   # monofractal only; no delta_h to compare
print("implementations:", ", ".join(ad), flush=True)

sigs = []
for s in range(1, a.subjects + 1):
    try:
        sigs += load_eegbci_channels(s, a.run, n_channels=a.channels)
    except Exception as e:
        print(f"  subject {s} unavailable: {type(e).__name__}", flush=True)
print(f"{len(sigs)} channels from {a.subjects} subjects", flush=True)

df = prevalence_table(sigs, ad, SC, Q)
df.to_csv("results/prevalence.csv", index=False)

res = summarize_prevalence(df)
print("\n=== delta_h across the cohort, by implementation ===")
print(res["by_package"].to_string())
print("\n=== association between flat-run content and delta_h ===")
print(res["association"].to_string())
print("\n=== channel flat-run content ===")
ch = df.drop_duplicates("signal")
print(f"  channels with any flat run >= 16 samples: {int((ch.flat_runs>0).sum())}/{len(ch)}"
      f" ({100*np.mean(ch.flat_runs>0):.1f}%)")
print(f"  median flat fraction: {ch.flat_fraction.median():.5f}   max: {ch.flat_fraction.max():.5f}")

"""Second-corpus run: PhysioNet Sleep-EDF."""
import warnings; warnings.filterwarnings("ignore")
import argparse
import numpy as np, pandas as pd
from benchmark.adapters import available_adapters
from benchmark.prevalence import prevalence_table, summarize_prevalence
from benchmark.signals import load_sleepedf_epochs

SC = np.array([16,24,32,48,64,96,128,192,256])
Q  = np.array([-5.,-3.,-2.,-1.,1.,2.,3.,5.])

ap = argparse.ArgumentParser()
ap.add_argument("--subjects", type=int, default=8)
ap.add_argument("--epochs", type=int, default=5)
a = ap.parse_args()

ad = available_adapters(); ad.pop("nolds", None)
print("implementations:", ", ".join(ad), flush=True)

sigs = []
for s in range(a.subjects):
    try:
        sigs += load_sleepedf_epochs(subject=s, n_epochs=a.epochs)
    except Exception as e:
        print(f"  subject {s} unavailable: {type(e).__name__}", flush=True)
print(f"{len(sigs)} epochs from Sleep-EDF", flush=True)

df = prevalence_table(sigs, ad, SC, Q)
df["corpus"] = "sleepedf"
df.to_csv("results/prevalence_sleepedf.csv", index=False)

res = summarize_prevalence(df)
print("\n=== Sleep-EDF: delta_h by implementation ===")
print(res["by_package"].to_string())
print("\n=== association with flat-run content ===")
print(res["association"].to_string())
ch = df.drop_duplicates("signal")
print(f"\nepochs with any flat run >= 16 samples: {int((ch.flat_runs>0).sum())}/{len(ch)}"
      f" ({100*np.mean(ch.flat_runs>0):.1f}%)")
print(f"median flat fraction: {ch.flat_fraction.median():.5f}  max: {ch.flat_fraction.max():.5f}")
print("\n=== delta_h > 1 split by flat-run presence ===")
for pkg, s in df.groupby("package"):
    s = s.dropna(subset=["delta_h"]); hf = s.flat_runs > 0
    a_ = 100*np.mean(s[hf].delta_h > 1) if hf.any() else float('nan')
    b_ = 100*np.mean(s[~hf].delta_h > 1) if (~hf).any() else float('nan')
    print(f"  {pkg:<18} with flat: {a_:5.1f}%   without: {b_:5.1f}%")

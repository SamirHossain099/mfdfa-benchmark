"""Scale-invariance test on the second corpus."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from benchmark.adapters import available_adapters
from benchmark.invariance import invariance_table, MULTIPLIERS
from benchmark.signals import load_sleepedf_epochs, load_eegbci_channels

SC = np.array([16,24,32,48,64,96,128,192,256])
Q  = np.array([-5.,-3.,-2.,-1.,1.,2.,3.,5.])
ad = available_adapters()
print("implementations:", ", ".join(ad), flush=True)

sigs  = [(f"sleepedf::{n}", x) for n, x in load_sleepedf_epochs(0, n_epochs=2)]
sigs += [(f"eegbci::{n}", x) for n, x in load_eegbci_channels(1, 4, n_channels=2)]

df = invariance_table(sigs, ad, SC, Q, verbose=False)
df.to_csv("results/invariance_two_corpora.csv", index=False)
df["corpus"] = np.where(df.signal.str.startswith("sleepedf"), "Sleep-EDF", "EEGBCI")

print("\n=== worst |h(q) drift| under rescaling, by corpus ===")
print(df.pivot_table(index="corpus", columns="package",
                     values="max_abs_drift", aggfunc="max").round(4).to_string())
print("\n=== % of (signal, multiplier) combinations that failed outright ===")
print((100*df.pivot_table(index="corpus", columns="package",
                          values="failed", aggfunc="mean")).round(1).to_string())

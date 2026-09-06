"""Out-of-process worker for `fathon`.

`fathon`'s published wheels are compiled against the NumPy 1.x ABI and fail to
import under NumPy 2, so it is run in its own interpreter. This script is
invoked as::

    <numpy1-python> fathon_worker.py <signal.npy> <scales_csv> <q_csv>

and prints ``{"hq": [...]}`` to stdout.
"""

import json
import sys

import numpy as np


def main() -> int:
    sig_path, scales_csv, q_csv = sys.argv[1], sys.argv[2], sys.argv[3]
    x = np.load(sig_path)
    # fathon's Cython layer requires int64; numpy's default int is int32 on Windows.
    scales = np.array([int(s) for s in scales_csv.split(",")], dtype=np.int64)
    q = np.array([float(v) for v in q_csv.split(",")], dtype=float)

    import fathon
    from fathon import fathonUtils as fu

    profile = fu.toAggregated(np.asarray(x, dtype=float))
    model = fathon.MFDFA(profile)
    # fathon wants a window vector; pass our scale grid directly.
    model.computeFlucVec(scales, qList=q, polOrd=1)
    hq, _intercepts = model.fitFlucVec()

    json.dump({"hq": np.asarray(hq, dtype=float).ravel().tolist()}, sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

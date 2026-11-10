"""Step 4 - sanity-check preprocessing outputs."""
import sys

import numpy as np
from validation import ROOT, load_config

cfg = load_config()
proc = ROOT / cfg["paths"]["processed_dir"]
Xtr, Xte = np.load(proc / "X_train_final.npy"), np.load(proc / "X_test_final.npy")
ytr, yte = np.load(proc / "y_train.npy"), np.load(proc / "y_test.npy")
errors = []
if Xtr.shape[0] != len(ytr) or Xte.shape[0] != len(yte):
    errors.append("row count mismatch between X and y")
if Xtr.shape[1] != Xte.shape[1]:
    errors.append("train/test feature count mismatch")
if np.isnan(Xtr).any() or np.isnan(Xte).any():
    errors.append("NaN values after preprocessing")
if np.isnan(ytr).any() or np.isnan(yte).any():
    errors.append("NaN in target")
if errors:
    print("[04] FAILED:", errors)
    sys.exit(1)
print("[04] preprocessing outputs valid")

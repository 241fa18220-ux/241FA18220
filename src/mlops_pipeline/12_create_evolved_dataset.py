"""Simulate evolved data: a later COVID wave (more cases, more deaths, faster growth)."""
import sys
from pathlib import Path

import pandas as pd
from validation import ROOT, clean_dataframe, features, load_config, path

cfg = load_config()
df = clean_dataframe(pd.read_csv(ROOT / cfg["paths"]["raw_data"]), cfg)
ev = df.copy()
for c, k in {"Confirmed": 1.6, "Deaths": 1.5, "Recovered": 1.35, "New cases": 2.0,
             "New deaths": 1.8, "New recovered": 1.6, "Confirmed last week": 1.45}.items():
    ev[c] = (ev[c] * k).round().astype(int)
ev["Deaths / 100 Cases"] = (100 * ev["Deaths"] / ev["Confirmed"].replace(0, pd.NA)).astype(float).fillna(0).round(2)
ev[cfg["id_column"]] = ev[cfg["id_column"]] + " (wave2)"
d = cfg["paths"]["drift_dir"]
ev[[cfg["id_column"]] + features(cfg)].to_csv(path(f"{d}/evolved_data.csv"), index=False)
ev[[cfg["id_column"]] + features(cfg) + [cfg["target"]]].to_csv(path(f"{d}/evolved_labeled_data.csv"), index=False)
print("[12] evolved datasets written to", d)

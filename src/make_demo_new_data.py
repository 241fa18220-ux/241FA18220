"""Create data/raw/new_countries.csv (no target column) for inference demos.
Takes 25 countries from the real file and perturbs the numbers by +/-10%. Replace with real new data anytime."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "mlops_pipeline"))
from validation import ROOT, clean_dataframe, features, load_config  # noqa: E402

cfg = load_config()
rng = np.random.default_rng(7)
df = clean_dataframe(pd.read_csv(ROOT / cfg["paths"]["raw_data"]), cfg).sample(25, random_state=7)
for c in ["Confirmed", "Deaths", "New cases", "New deaths", "New recovered", "Confirmed last week"]:
    df[c] = (df[c] * rng.uniform(0.9, 1.1, len(df))).round().astype(int)
df["Deaths / 100 Cases"] = (100 * df["Deaths"] / df["Confirmed"].replace(0, np.nan)).fillna(0).round(2)
out = ROOT / cfg["paths"]["new_data"]
out.parent.mkdir(parents=True, exist_ok=True)
df[[cfg["id_column"]] + features(cfg)].to_csv(out, index=False)
print("wrote", out)

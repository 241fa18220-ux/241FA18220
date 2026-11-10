"""Step 2 - validate raw data schema & quality."""
import sys

import pandas as pd
from validation import ROOT, clean_dataframe, load_config, path, validate_schema

cfg = load_config()
df = clean_dataframe(pd.read_csv(ROOT / cfg["paths"]["raw_data"]), cfg)
issues = validate_schema(df, cfg)
pd.DataFrame({"column": df.columns, "dtype": df.dtypes.astype(str).values,
              "nulls": df.isna().sum().values, "unique": df.nunique().values}
             ).to_csv(path("artifacts/baseline_validation.csv"), index=False)
if issues:
    print("[02] VALIDATION FAILED:", issues)
    sys.exit(1)
print(f"[02] data valid | mean Recovered = {df[cfg['target']].mean():,.0f}")

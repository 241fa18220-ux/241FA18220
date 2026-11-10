"""Step 1 - load raw country-wise COVID data."""
import pandas as pd
from validation import ROOT, load_config, write_json

cfg = load_config()
df = pd.read_csv(ROOT / cfg["paths"]["raw_data"])
write_json("artifacts/data_load_report.json", {
    "file": cfg["paths"]["raw_data"], "rows": len(df), "columns": list(df.columns),
    "null_counts": df.isna().sum().to_dict(), "status": "ok" if len(df) else "empty"})
print(f"[01] loaded {len(df)} countries, {df.shape[1]} columns")

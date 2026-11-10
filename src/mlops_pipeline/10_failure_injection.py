"""Step 10 - failure simulation: corrupt data and confirm validation catches it."""
import pandas as pd
from validation import ROOT, clean_dataframe, load_config, path, validate_schema, write_json

cfg = load_config()
df = pd.read_csv(ROOT / cfg["paths"]["raw_data"])
neg = df.copy(); neg.loc[0, "Confirmed"] = -5
big = df.copy(); big.loc[1, "Recovered"] = big.loc[1, "Confirmed"] * 3 + 10
cases = {
    "missing_column": df.drop(columns=["WHO Region"]),
    "negative_counts": neg,
    "recovered_exceeds_confirmed": big,
    "duplicate_countries": pd.concat([df, df.head(5)]),
}
rows = []
for name, bad in cases.items():
    issues = validate_schema(clean_dataframe(bad, cfg), cfg)
    rows.append({"case": name, "detected": bool(issues), "issues": str(issues)})
pd.DataFrame(rows).to_csv(path("artifacts/corrupted_simulation_errors.csv"), index=False)
write_json("artifacts/failure_simulation_report.json", {"all_detected": all(r["detected"] for r in rows), "cases": rows})
print("[10]", {r["case"]: r["detected"] for r in rows})

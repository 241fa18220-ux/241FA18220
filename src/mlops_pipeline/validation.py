"""Shared helpers used by every pipeline step (config, paths, cleaning, validation)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_config() -> dict:
    return yaml.safe_load((ROOT / "config.yaml").read_text())


def features(cfg) -> list:
    return cfg["numeric_features"] + cfg["categorical_features"]


def path(rel: str) -> Path:
    p = ROOT / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def write_json(rel: str, obj) -> Path:
    p = path(rel)
    p.write_text(json.dumps(obj, indent=2, default=str))
    return p


def read_json(rel: str):
    return json.loads((ROOT / rel).read_text())


def clean_dataframe(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Strip names/strings, coerce numerics, drop columns the notebook drops (unused + leakage)."""
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    for col in cfg["numeric_features"] + [cfg["target"]]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in cfg["categorical_features"] + [cfg["id_column"]]:
        if col in df.columns:
            df[col] = df[col].astype("string").str.strip().astype(object)
    drop = [c for c in cfg["drop_columns"] + cfg["leakage_columns"] if c in df.columns]
    return df.drop(columns=drop)


def validate_schema(df: pd.DataFrame, cfg: dict, need_target: bool = True) -> list:
    """Return a list of issue dicts (empty list == valid)."""
    issues = []
    required = [cfg["id_column"]] + features(cfg) + ([cfg["target"]] if need_target else [])
    missing = [c for c in required if c not in df.columns]
    if missing:
        return [{"check": "missing_columns", "detail": missing}]
    if df.duplicated().sum():
        issues.append({"check": "duplicate_rows", "detail": int(df.duplicated().sum())})
    if df[cfg["id_column"]].duplicated().any():
        issues.append({"check": "duplicate_countries", "detail": int(df[cfg["id_column"]].duplicated().sum())})
    count_cols = ["Confirmed", "Deaths", "New cases", "New deaths", "New recovered", "Confirmed last week"]
    if need_target:
        count_cols.append(cfg["target"])
    neg = [c for c in count_cols if c in df.columns and (df[c] < 0).any()]
    if neg:
        issues.append({"check": "negative_counts", "detail": neg})
    if (df["Deaths"] > df["Confirmed"]).any():
        issues.append({"check": "deaths_exceed_confirmed", "detail": int((df["Deaths"] > df["Confirmed"]).sum())})
    if need_target:
        t = cfg["target"]
        if df[t].isna().any():
            issues.append({"check": "null_target", "detail": int(df[t].isna().sum())})
        if (df[t] > df["Confirmed"]).any():
            issues.append({"check": "recovered_exceeds_confirmed", "detail": int((df[t] > df["Confirmed"]).sum())})
    return issues

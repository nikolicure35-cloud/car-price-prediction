"""Derived features used by both training and inference."""
from __future__ import annotations
from datetime import date
import pandas as pd


def add_features(df: pd.DataFrame, reference_year: int | None = None) -> pd.DataFrame:
    out = df.copy()
    reference_year = reference_year or date.today().year
    if "year" in out:
        out["car_age"] = (reference_year - pd.to_numeric(out["year"], errors="coerce")).clip(lower=0)
    if "volume_cm3" in out:
        out["engine_volume_liters"] = pd.to_numeric(out["volume_cm3"], errors="coerce") / 1000
    if "mileage_km" in out:
        age = out.get("car_age", pd.Series(1, index=out.index)).clip(lower=1)
        out["mileage_per_year"] = pd.to_numeric(out["mileage_km"], errors="coerce") / age
        out["is_high_mileage"] = (pd.to_numeric(out["mileage_km"], errors="coerce") >= 250_000).astype("int64")
    if {"make", "model"}.issubset(out.columns):
        out["brand_model"] = out["make"].fillna("unknown").astype(str) + "__" + out["model"].fillna("unknown").astype(str)
    return out

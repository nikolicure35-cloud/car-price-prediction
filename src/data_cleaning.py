"""Basic validation and normalization for the raw cars dataset."""
from __future__ import annotations
import pandas as pd

COLUMN_MAP = {
    "make": "make", "model": "model", "priceUSD": "price_usd", "year": "year",
    "condition": "condition", "mileage(kilometers)": "mileage_km",
    "fuel_type": "fuel_type", "volume(cm3)": "volume_cm3", "color": "color",
    "transmission": "transmission", "drive_unit": "drive_unit", "segment": "segment",
}


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy; retain valid rows with incomplete optional fields."""
    df = df.rename(columns=COLUMN_MAP).copy()
    if "price_usd" not in df:
        raise ValueError("Dataset must include priceUSD")
    df["price_usd"] = pd.to_numeric(df["price_usd"], errors="coerce")
    for col in ("year", "mileage_km", "volume_cm3"):
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["price_usd"])
    # A non-positive price cannot represent a usable sale listing.
    df = df[df["price_usd"] > 0]
    # Treat implausible numeric measurements as missing; imputers handle them later.
    if "year" in df:
        df.loc[~df["year"].between(1950, 2026), "year"] = float("nan")
    if "mileage_km" in df:
        df.loc[~df["mileage_km"].between(0, 1_000_000), "mileage_km"] = float("nan")
    if "volume_cm3" in df:
        df.loc[~df["volume_cm3"].between(500, 10_000), "volume_cm3"] = float("nan")
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip().str.lower().replace({"": pd.NA, "nan": pd.NA, "none": pd.NA})
    return df.reset_index(drop=True)


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data/cars.csv")
    p.add_argument("--output", default="data/cars_clean.csv")
    a = p.parse_args()
    data = clean_data(pd.read_csv(a.input))
    data.to_csv(a.output, index=False)
    print(f"Saved {len(data):,} cleaned rows to {a.output}")

if __name__ == "__main__": main()

"""Load the final saved pipeline and predict price for a car dictionary."""
from __future__ import annotations
import joblib
import pandas as pd
try:
    from .feature_engineering import add_features
except ImportError:
    from feature_engineering import add_features


def predict_price(car: dict, model_path="models/car_price_model.joblib") -> float:
    bundle = joblib.load(model_path)
    # Accept original CSV names or normalized names, as long as a target is omitted.
    row = pd.DataFrame([car])
    row = row.rename(columns={"mileage(kilometers)":"mileage_km", "volume(cm3)":"volume_cm3"})
    row = add_features(row, reference_year=bundle["reference_year"])
    row = row.reindex(columns=bundle["feature_columns"])
    return float(bundle["pipeline"].predict(row)[0])

if __name__ == "__main__":
    example={"make":"volkswagen","model":"golf","year":2014,"condition":"with mileage","mileage_km":180000,"fuel_type":"diesel","volume_cm3":1600,"color":"black","transmission":"mechanics","drive_unit":"front-wheel drive","segment":"c"}
    print(f"Estimated price: ${predict_price(example):,.0f}")

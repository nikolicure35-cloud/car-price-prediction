"""Reusable train/test split and sklearn preprocessing pipeline."""
from __future__ import annotations
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "price_usd"
NUMERIC_FEATURES = ["year", "mileage_km", "volume_cm3", "car_age", "engine_volume_liters", "mileage_per_year", "is_high_mileage"]
CATEGORICAL_FEATURES = ["make", "model", "condition", "fuel_type", "color", "transmission", "drive_unit", "segment", "brand_model"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def make_preprocessor():
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median", add_indicator=True)), ("scaler", StandardScaler())])
    categorical = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=5))])
    return ColumnTransformer([("numeric", numeric, NUMERIC_FEATURES), ("categorical", categorical, CATEGORICAL_FEATURES)], remainder="drop")

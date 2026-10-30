"""Train and compare regressors using one reproducible hold-out split."""
from __future__ import annotations
import json
from pathlib import Path
from datetime import date
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from data_cleaning import clean_data
from feature_engineering import add_features
from data_preprocessing import FEATURES, TARGET, make_preprocessor

SEED = 42
NUMERIC = {"year", "mileage_km", "volume_cm3", "car_age", "engine_volume_liters", "mileage_per_year", "is_high_mileage"}

def metric_row(name, y, pred):
    mse = mean_squared_error(y, pred)
    return {"model": name, "MAE": mean_absolute_error(y, pred), "MSE": mse, "RMSE": mse ** 0.5, "R2": r2_score(y, pred)}

def candidates():
    return {
        "Ridge Regression": Ridge(alpha=10.0),
        "Decision Tree": DecisionTreeRegressor(max_depth=20, min_samples_leaf=8, random_state=SEED),
        "Random Forest": RandomForestRegressor(n_estimators=50, max_depth=24, min_samples_leaf=4, max_features=0.8, n_jobs=-1, random_state=SEED),
        "Extra Trees": ExtraTreesRegressor(n_estimators=50, max_depth=24, min_samples_leaf=4, max_features=0.8, n_jobs=-1, random_state=SEED),
        "HistGradientBoosting": HistGradientBoostingRegressor(max_iter=100, learning_rate=0.08, max_leaf_nodes=25, l2_regularization=1.0, random_state=SEED),
    }

def train(input_path="data/cars.csv", model_dir="models", report_dir="reports"):
    df = add_features(clean_data(pd.read_csv(input_path)))
    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)
    rows, fitted = [], {}
    for name, estimator in candidates().items():
        preprocessor = make_preprocessor()
        if name == "HistGradientBoosting":
            from sklearn.compose import ColumnTransformer
            from sklearn.impute import SimpleImputer
            from sklearn.preprocessing import OneHotEncoder, StandardScaler
            from sklearn.pipeline import Pipeline as SkPipeline
            cat = [c for c in FEATURES if c not in NUMERIC]
            num = [c for c in FEATURES if c in NUMERIC]
            preprocessor = ColumnTransformer([
                ("numeric", SkPipeline([("imputer", SimpleImputer(strategy="median", add_indicator=True)), ("scaler", StandardScaler())]), num),
                ("categorical", SkPipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=5, sparse_output=False))]), cat),
            ], sparse_threshold=0)
        pipe = Pipeline([("preprocessor", preprocessor), ("regressor", estimator)])
        pipe.fit(X_train, y_train)
        row = metric_row(name, y_test, pipe.predict(X_test))
        rows.append(row); fitted[name] = pipe
        print(f"{name}: MAE=${row['MAE']:,.0f}, RMSE=${row['RMSE']:,.0f}, R2={row['R2']:.3f}", flush=True)
    results = pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)
    best_name = results.loc[0, "model"]
    model_path, reports = Path(model_dir), Path(report_dir)
    model_path.mkdir(parents=True, exist_ok=True); reports.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": fitted[best_name], "feature_columns": FEATURES, "model_name": best_name, "reference_year": date.today().year, "target": TARGET}, model_path / "car_price_model.joblib", compress=('lzma', 9))
    results.to_csv(reports / "model_comparison.csv", index=False)
    pred = fitted[best_name].predict(X_test)
    pd.DataFrame({"actual_price_usd": y_test.values, "predicted_price_usd": pred, "absolute_error_usd": abs(y_test.values-pred)}).head(20).to_csv(reports / "sample_predictions.csv", index=False)
    (reports / "training_summary.json").write_text(json.dumps({"rows": len(df), "train_rows": len(X_train), "test_rows": len(X_test), "best_model": best_name, "best_metrics": rows[[r["model"] for r in rows].index(best_name)]}, indent=2), encoding="utf-8")
    print(f"Selected {best_name}; saved {model_path / 'car_price_model.joblib'}", flush=True)
    return results

if __name__ == "__main__":
    import argparse
    p=argparse.ArgumentParser(); p.add_argument("--input",default="data/cars.csv"); p.add_argument("--model-dir",default="models"); p.add_argument("--report-dir",default="reports"); a=p.parse_args()
    train(a.input,a.model_dir,a.report_dir)

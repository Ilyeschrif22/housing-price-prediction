"""
Train and save a real-estate price prediction model.

Dataset schema (data.csv):
category, room_count, bathroom_count, size, type, price, city, region, log_price
"""

import os
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


def load_and_clean_data(csv_path: str) -> pd.DataFrame:
    data = pd.read_csv(csv_path)

    # Replace sentinel -1 values with NaN, then drop rows with missing required fields
    data = data.replace(-1, np.nan)
    data = data.drop_duplicates()

    # log_price is redundant for training/prediction
    if "log_price" in data.columns:
        data = data.drop(columns=["log_price"])

    required = [
        "category",
        "room_count",
        "bathroom_count",
        "size",
        "type",
        "price",
        "city",
        "region",
    ]
    data = data.dropna(subset=required).copy()

    # Basic sanity: keep non-negative sizes and prices
    data = data[(data["size"] > 0) & (data["price"] > 0)].copy()

    # Keep only known transaction types
    data["type"] = data["type"].astype(str).str.strip()
    data = data[data["type"].isin(["À Vendre", "À Louer"])].copy()

    return data


def fit_encoders(data: pd.DataFrame, cat_cols):
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        le.fit(data[col].astype(str).str.strip())
        encoders[col] = le
    return encoders


def encode_categories(
    data: pd.DataFrame, encoders, cat_cols
) -> pd.DataFrame:
    out = data.copy()
    for col in cat_cols:
        out[col] = out[col].astype(str).str.strip()
        out[col + "_enc"] = encoders[col].transform(out[col])
    return out


def main() -> None:
    csv_path = "data.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            "data.csv not found. Place it next to train_model.py (dataset2/data.csv)."
        )

    print("Loading data...")
    data = load_and_clean_data(csv_path)
    print(f"Rows after cleaning: {len(data)}")

    # We train one model per transaction type ("À Vendre" vs "À Louer") because
    # the target scale differs drastically (sale vs rent).
    # 'type' stays an input for routing, not a numeric feature.
    cat_cols = ["category", "city", "region"]
    encoders = fit_encoders(data, cat_cols)
    data = encode_categories(data, encoders, cat_cols)

    feature_cols = [
        "room_count",
        "bathroom_count",
        "size",
        "category_enc",
        "city_enc",
        "region_enc",
    ]

    X = data[feature_cols].copy()
    y = data["price"].copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)

    with open(os.path.join(models_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    with open(os.path.join(models_dir, "feature_cols.pkl"), "wb") as f:
        pickle.dump(feature_cols, f)

    with open(os.path.join(models_dir, "label_encoders.pkl"), "wb") as f:
        pickle.dump(encoders, f)

    # Persist dataset stats + per-type model metrics for the API
    stats = {
        "rows": int(len(data)),
        "categorical_levels": {k: int(data[k].nunique()) for k in cat_cols},
        "by_type": {},
    }

    def _train_one(type_value: str, slug: str) -> None:
        mask = data["type"] == type_value
        if int(mask.sum()) < 50:
            raise RuntimeError(f"Not enough rows for training '{type_value}'")

        X_t = X_scaled[mask.values]
        y_t = y[mask].astype(float).values

        # Stabilize training by predicting log(price) then inverse-transforming
        y_log = np.log1p(y_t)

        X_train, X_test, y_train, y_test = train_test_split(
            X_t, y_log, test_size=0.2, random_state=42
        )

        print(f"Training model for '{type_value}'...")
        model = RandomForestRegressor(
            n_estimators=400,
            random_state=42,
            n_jobs=-1,
            max_depth=None,
            min_samples_leaf=2,
        )
        model.fit(X_train, y_train)

        y_pred_log = model.predict(X_test)
        y_pred = np.expm1(y_pred_log)
        y_true = np.expm1(y_test)

        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mae = float(mean_absolute_error(y_true, y_pred))
        r2 = float(r2_score(y_true, y_pred))
        print(f"  Holdout: RMSE={rmse:.2f} MAE={mae:.2f} R2={r2:.4f}")

        with open(os.path.join(models_dir, f"price_model_{slug}.pkl"), "wb") as f:
            pickle.dump(model, f)

        stats["by_type"][slug] = {
            "label": type_value,
            "rows": int(mask.sum()),
            "price_min": float(np.min(y_t)),
            "price_max": float(np.max(y_t)),
            "price_mean": float(np.mean(y_t)),
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
        }

    _train_one("À Vendre", "vendre")
    _train_one("À Louer", "louer")

    with open(os.path.join(models_dir, "stats.pkl"), "wb") as f:
        pickle.dump(stats, f)

    print(f"Saved artifacts to {models_dir}/")


if __name__ == "__main__":
    main()

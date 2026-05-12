"""
Flask API for real-estate price prediction (regression).
"""

import os
import pickle

import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS


def _load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


app = Flask(__name__)
CORS(app)

models_dir = "models"
if not os.path.exists(models_dir):
    raise RuntimeError("Models directory not found. Run 'python train_model.py' first.")

model_vendre = _load_pickle(os.path.join(models_dir, "price_model_vendre.pkl"))
model_louer = _load_pickle(os.path.join(models_dir, "price_model_louer.pkl"))
scaler = _load_pickle(os.path.join(models_dir, "scaler.pkl"))
feature_cols = _load_pickle(os.path.join(models_dir, "feature_cols.pkl"))
label_encoders = _load_pickle(os.path.join(models_dir, "label_encoders.pkl"))
stats = _load_pickle(os.path.join(models_dir, "stats.pkl"))


def _encode_or_400(col: str, value: str):
    if col not in label_encoders:
        return None
    le = label_encoders[col]
    value_norm = str(value).strip()
    if value_norm not in le.classes_:
        return None
    return int(le.transform([value_norm])[0])


def _bad_request(msg: str, details=None):
    payload = {"error": msg}
    if details:
        payload["details"] = details
    return jsonify(payload), 400


@app.route("/health", methods=["GET"])
def health():
    return jsonify(
        {
            "status": "healthy",
            "model": "RandomForestRegressor",
            "rows": stats.get("rows"),
        }
    )


@app.route("/stats", methods=["GET"])
def api_stats():
    return jsonify(
        {
            "rows": stats.get("rows"),
            "categorical_levels": stats.get("categorical_levels", {}),
            "by_type": stats.get("by_type", {}),
        }
    )


@app.route("/schema", methods=["GET"])
def schema():
    return jsonify(
        {
            "endpoint": "POST /predict",
            "required_fields": {
                "room_count": "number",
                "bathroom_count": "number",
                "size": "number",
                "category": "string",
                "type": "string",
                "city": "string",
                "region": "string",
            },
            "allowed_type_values": ["À Vendre", "À Louer"],
        }
    )


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(silent=True) or {}

    required = [
        "room_count",
        "bathroom_count",
        "size",
        "category",
        "type",
        "city",
        "region",
    ]
    missing = [k for k in required if k not in payload]
    if missing:
        return _bad_request("Missing required fields", {"missing": missing})

    try:
        room_count = float(payload["room_count"])
        bathroom_count = float(payload["bathroom_count"])
        size = float(payload["size"])
    except Exception:
        return _bad_request("room_count, bathroom_count, and size must be numbers")

    if size <= 0:
        return _bad_request("size must be > 0")

    category_enc = _encode_or_400("category", payload["category"])
    if category_enc is None:
        return _bad_request(
            "Unknown category",
            {"allowed": sorted(map(str, label_encoders["category"].classes_.tolist()))},
        )

    type_value = str(payload["type"]).strip()
    if type_value not in ["À Vendre", "À Louer"]:
        return _bad_request("Unknown type", {"allowed": ["À Vendre", "À Louer"]})

    city_enc = _encode_or_400("city", payload["city"])
    if city_enc is None:
        return _bad_request(
            "Unknown city",
            {"allowed": sorted(map(str, label_encoders["city"].classes_.tolist()))},
        )

    region_enc = _encode_or_400("region", payload["region"])
    if region_enc is None:
        return _bad_request(
            "Unknown region",
            {"allowed": sorted(map(str, label_encoders["region"].classes_.tolist()))},
        )

    row = np.array(
        [
            room_count,
            bathroom_count,
            size,
            category_enc,
            city_enc,
            region_enc,
        ],
        dtype=float,
    ).reshape(1, -1)

    X_scaled = scaler.transform(row)
    if type_value == "À Louer":
        model = model_louer
        slug = "louer"
    else:
        model = model_vendre
        slug = "vendre"

    # Models predict log(price), so we inverse-transform
    pred_log = float(model.predict(X_scaled)[0])
    pred = float(np.expm1(pred_log))

    # Ensure non-negative prediction
    pred = max(0.0, pred)

    return jsonify(
        {
            "success": True,
            "input": {
                "room_count": room_count,
                "bathroom_count": bathroom_count,
                "size": size,
                "category": str(payload["category"]).strip(),
                "type": str(payload["type"]).strip(),
                "city": str(payload["city"]).strip(),
                "region": str(payload["region"]).strip(),
            },
            "model_used": slug,
            "prediction": {"price_tnd": pred},
        }
    )


@app.route("/", methods=["GET"])
def index():
    return jsonify(
        {
            "name": "Real Estate Price Prediction API",
            "version": "1.0.0",
            "endpoints": {
                "GET /health": "Health check",
                "GET /stats": "Dataset and model metrics",
                "GET /schema": "Request schema for /predict",
                "POST /predict": "Predict price",
                "GET /": "This message",
            },
            "example_request": {
                "room_count": 2,
                "bathroom_count": 1,
                "size": 80,
                "category": "Appartements",
                "type": "À Vendre",
                "city": "Ariana",
                "region": "Raoued",
            },
        }
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)

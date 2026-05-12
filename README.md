# Real Estate Price Prediction System — Railway Deployment

This project exposes a production-ready Flask API that predicts the numeric price (TND) of a property from its characteristics.

The dataset contains both sale ("À Vendre") and rent ("À Louer") rows, so this project trains and serves two separate models and routes automatically based on the `type` field.

## Prerequisites

- Python 3.8+
- `data.csv` in the project directory (already included in `dataset2/data.csv`)

## Local setup

```bash
cd dataset2
pip install -r requirements.txt
python train_model.py
python app.py
```

The API runs at `http://localhost:5000`.

## API

- `GET /health` health check
- `GET /stats` dataset stats and holdout metrics
- `GET /schema` required request fields
- `POST /predict` predict a price

## Quick curl tests (Railway) (1, 2, 3, 5)

```bash
# 1) Health


# 5) Predict (À Vendre)
curl -s -X POST https://housing-price-prediction-production-8801.up.railway.app/predict -H "Content-Type: application/json" \
  -d '{"room_count":2,"bathroom_count":1,"size":95,"category":"Appartements","type":"À Vendre","city":"Ariana","region":"La Soukra"}'

# Rent example 1 (À Louer) - Ariana / La Soukra
curl -s -X POST https://housing-price-prediction-production-8801.up.railway.app/predict -H "Content-Type: application/json" \
  -d '{"room_count":2,"bathroom_count":1,"size":80,"category":"Appartements","type":"À Louer","city":"Ariana","region":"La Soukra"}'

# Rent example 2 (À Louer) - Sousse / Sahloul
curl -s -X POST https://housing-price-prediction-production-8801.up.railway.app/predict -H "Content-Type: application/json" \
  -d '{"room_count":2,"bathroom_count":1,"size":85,"category":"Appartements","type":"À Louer","city":"Sousse","region":"Sahloul"}'

# Rent example 3 (À Louer) - Sousse / Hammam Sousse
curl -s -X POST https://housing-price-prediction-production-8801.up.railway.app/predict -H "Content-Type: application/json" \
  -d '{"room_count":3,"bathroom_count":2,"size":120,"category":"Maisons et Villas","type":"À Louer","city":"Sousse","region":"Hammam Sousse"}'

# Rent example 4 (À Louer) - Tunis / La Marsa
curl -s -X POST https://housing-price-prediction-production-8801.up.railway.app/predict -H "Content-Type: application/json" \
  -d '{"room_count":2,"bathroom_count":1,"size":90,"category":"Appartements","type":"À Louer","city":"Tunis","region":"La Marsa"}'
```

### Example request

```bash
curl -X POST http://localhost:5000/predict \\
  -H "Content-Type: application/json" \\
  -d '{
    "room_count": 2,
    "bathroom_count": 1,
    "size": 80,
    "category": "Appartements",
    "type": "À Vendre",
    "city": "Ariana",
    "region": "Raoued"
  }'
```

## Railway deploy

This folder includes `Procfile` and `railway.json`.

- Build command: installs dependencies from `requirements.txt`
- Release step: `python train_model.py`
- Start command: `gunicorn app:app`

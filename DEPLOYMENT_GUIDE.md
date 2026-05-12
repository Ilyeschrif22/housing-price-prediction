# Deployment Guide (Railway)

This folder is deployable as a standalone Railway service.

## What Railway runs

- Release step: `python train_model.py` (creates `models/`)
- Start command: `gunicorn app:app`

## Notes

- `data.csv` must be in the project root (this repo includes `dataset2/data.csv`).
- Railway sets `PORT` automatically; the app reads it.


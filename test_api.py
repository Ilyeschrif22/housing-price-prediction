"""
Test script for the Real Estate Price Prediction API
Usage:
  python test_api.py
  python test_api.py https://your-railway-url.railway.app
"""

import json
import sys

import requests


def main() -> int:
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:5000"

    print("=" * 70)
    print("Testing Real Estate Price Prediction API")
    print(f"Base URL: {base_url}")
    print("=" * 70)

    print("\n1) Health check")
    try:
        r = requests.get(f"{base_url}/health", timeout=5)
        print(f"Status: {r.status_code}")
        print(json.dumps(r.json(), indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"Error: {e}")

    print("\n2) Stats")
    try:
        r = requests.get(f"{base_url}/stats", timeout=5)
        print(f"Status: {r.status_code}")
        print(json.dumps(r.json(), indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"Error: {e}")

    print("\n3) Schema")
    try:
        r = requests.get(f"{base_url}/schema", timeout=5)
        print(f"Status: {r.status_code}")
        print(json.dumps(r.json(), indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"Error: {e}")

    print("\n4) Predict")
    payload = {
        "room_count": 2,
        "bathroom_count": 1,
        "size": 80,
        "category": "Appartements",
        "type": "À Vendre",
        "city": "Ariana",
        "region": "Raoued",
    }
    print("Request payload:")
    print(json.dumps(payload, indent=2, ensure_ascii=False))

    try:
        r = requests.post(f"{base_url}/predict", json=payload, timeout=10)
        print(f"Status: {r.status_code}")
        if r.headers.get("content-type", "").startswith("application/json"):
            print(json.dumps(r.json(), indent=2, ensure_ascii=False))
        else:
            print(r.text)
    except Exception as e:
        print(f"Error: {e}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


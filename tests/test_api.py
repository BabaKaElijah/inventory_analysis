from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import app
import model_trainer


client = TestClient(app.app)


def _make_synthetic_df(rows: int = 50) -> pd.DataFrame:
    rng = random.Random(42)
    np.random.seed(42)

    stores = ["S001", "S002"]
    products = ["P001", "P002"]
    categories = ["Electronics", "Clothing", "Home"]
    regions = ["North", "South"]
    weather = ["Sunny", "Rainy"]
    seasons = ["Spring", "Summer"]

    data = []
    for _ in range(rows):
        data.append(
            {
                "Store ID": rng.choice(stores),
                "Product ID": rng.choice(products),
                "Category": rng.choice(categories),
                "Region": rng.choice(regions),
                "Inventory Level": rng.randint(50, 200),
                "Units Sold": rng.randint(20, 180),
                "Units Ordered": rng.randint(10, 200),
                "Demand Forecast": rng.uniform(50, 200),
                "Price": rng.uniform(10, 100),
                "Discount": rng.randint(0, 20),
                "Weather Condition": rng.choice(weather),
                "Holiday/Promotion": rng.randint(0, 1),
                "Competitor Pricing": rng.uniform(10, 120),
                "Seasonality": rng.choice(seasons),
                "day_of_week": rng.randint(0, 6),
                "month": rng.randint(1, 12),
                "day": rng.randint(1, 28),
                "is_weekend": rng.randint(0, 1),
            }
        )

    return pd.DataFrame(data)


@pytest.fixture(scope="session")
def ensure_model():
    model_path = Path("artifacts") / "models" / "best_model.pkl"
    if model_path.exists():
        return

    model_trainer.mlflow = None
    df = _make_synthetic_df()
    model_trainer.train_model(df, n_estimators=5, random_state=0)


def _sample_payload() -> dict:
    return {
        "Store_ID": "S001",
        "Product_ID": "P001",
        "Category": "Electronics",
        "Region": "North",
        "Inventory_Level": 120,
        "Units_Ordered": 80,
        "Demand_Forecast": 140.5,
        "Price": 49.99,
        "Discount": 10,
        "Weather_Condition": "Sunny",
        "Holiday_Promotion": 0,
        "Competitor_Pricing": 52.0,
        "Seasonality": "Summer",
        "day_of_week": 2,
        "month": 7,
        "day": 15,
        "is_weekend": 0,
    }


def test_predict_success(ensure_model):
    response = client.post("/predict", json=_sample_payload())
    assert response.status_code == 200
    body = response.json()
    assert "prediction" in body
    assert isinstance(body["prediction"], (int, float))


def test_predict_missing_model():
    model_path = Path("artifacts") / "models" / "best_model.pkl"
    tmp_path = None
    if model_path.exists():
        tmp_path = model_path.with_suffix(".bak")
        model_path.rename(tmp_path)

    try:
        response = client.post("/predict", json=_sample_payload())
        assert response.status_code == 404
    finally:
        if tmp_path and tmp_path.exists():
            tmp_path.rename(model_path)


def _has_explain_route() -> bool:
    return any(route.path == "/explain" for route in app.app.routes)


def test_explain_success(ensure_model):
    if not _has_explain_route():
        pytest.skip("/explain not available in current API")

    response = client.post("/explain", json=_sample_payload())
    assert response.status_code == 200
    body = response.json()
    assert "prediction" in body
    assert "contributions" in body
    assert isinstance(body["contributions"], list)


def test_explain_missing_model():
    if not _has_explain_route():
        pytest.skip("/explain not available in current API")

    model_path = Path("artifacts") / "models" / "best_model.pkl"
    tmp_path = None
    if model_path.exists():
        tmp_path = model_path.with_suffix(".bak")
        model_path.rename(tmp_path)

    try:
        response = client.post("/explain", json=_sample_payload())
        assert response.status_code == 404
    finally:
        if tmp_path and tmp_path.exists():
            tmp_path.rename(model_path)

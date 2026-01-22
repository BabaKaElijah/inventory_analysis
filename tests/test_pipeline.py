from pathlib import Path

import pandas as pd

from data_ingestion import ingest
from data_transformation import transform
from model_trainer import train_model


def test_pipeline_smoke():
    data_path = Path("data") / "retail_store_inventory.csv"
    df = ingest(data_path)
    assert not df.empty

    df = transform(df)
    for col in ["day_of_week", "month", "day", "is_weekend"]:
        assert col in df.columns

    sample = df.sample(n=500, random_state=42)
    model, mae = train_model(sample, n_estimators=10, random_state=42)
    assert model is not None
    assert mae >= 0


def test_model_regression_mae_threshold():
    data_path = Path("data") / "retail_store_inventory.csv"
    df = ingest(data_path)
    df = transform(df)

    sample = df.sample(n=1000, random_state=42)
    _, mae = train_model(sample, n_estimators=20, random_state=42)

    assert mae <= 50

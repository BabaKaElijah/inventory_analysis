from pathlib import Path

import os

from data_ingestion import ingest
from data_transformation import transform
from model_evaluation import evaluate_model
from model_trainer import train_model
from logger import get_logger


logger = get_logger(__name__)


def run() -> None:
    data_path = Path("data") / "retail_store_inventory.csv"
    df = ingest(data_path)
    df = transform(df)
    if os.getenv("FAST_TRAIN") == "1":
        df = df.sample(n=20000, random_state=42)
        model, mae, X_test, y_test = train_model(
            df, n_estimators=80, random_state=42, return_data=True
        )
    else:
        model, mae, X_test, y_test = train_model(df, return_data=True)
    evaluate_model(model, X_test, y_test)
    logger.info("Pipeline completed with MAE %s", mae)


if __name__ == "__main__":
    run()

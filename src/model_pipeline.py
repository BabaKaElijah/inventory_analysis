from pathlib import Path

from data_ingestion import ingest
from data_transformation import transform
from model_trainer import train_model
from logger import get_logger


logger = get_logger(__name__)


def run() -> None:
    data_path = Path("data") / "retail_store_inventory.csv"
    df = ingest(data_path)
    df = transform(df)
    _, mae = train_model(df)
    logger.info("Pipeline completed with MAE %s", mae)


if __name__ == "__main__":
    run()

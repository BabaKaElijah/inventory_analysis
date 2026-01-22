import pandas as pd
from pathlib import Path
from logger import get_logger


logger = get_logger(__name__)


def ingest(data_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    logger.info("Loaded raw data with shape %s", df.shape)
    return df

import pandas as pd
from logger import get_logger


logger = get_logger(__name__)


def transform(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df["day_of_week"] = df["Date"].dt.dayofweek
    df["month"] = df["Date"].dt.month
    df["day"] = df["Date"].dt.day
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    logger.info("Transformed data with shape %s", df.shape)
    return df

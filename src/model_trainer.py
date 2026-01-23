import json
import pickle
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor

from logger import get_logger


logger = get_logger(__name__)

try:
    import mlflow  # type: ignore
except ModuleNotFoundError:
    mlflow = None


def save_best_model(model, metric_name: str, metric_value: float) -> bool:
    artifacts_dir = Path("artifacts")
    models_dir = artifacts_dir / "models"
    metrics_dir = artifacts_dir / "metrics"
    models_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = metrics_dir / "best_metrics.json"
    best_value = None
    if metrics_path.exists():
        try:
            best_value = float(json.loads(metrics_path.read_text()).get(metric_name))
        except (ValueError, TypeError, json.JSONDecodeError):
            best_value = None

    is_better = best_value is None or metric_value < best_value
    if is_better:
        with (models_dir / "best_model.pkl").open("wb") as f:
            pickle.dump(model, f)
        metrics_path.write_text(
            json.dumps({metric_name: float(metric_value)}, indent=2),
            encoding="utf-8",
        )
    return is_better


def train_model(
    df: pd.DataFrame,
    n_estimators: int = 200,
    random_state: int = 42,
    return_data: bool = False,
):
    target = "Units Sold"
    features = [
        "Store ID",
        "Product ID",
        "Category",
        "Region",
        "Inventory Level",
        "Units Ordered",
        "Demand Forecast",
        "Price",
        "Discount",
        "Weather Condition",
        "Holiday/Promotion",
        "Competitor Pricing",
        "Seasonality",
        "day_of_week",
        "month",
        "day",
        "is_weekend",
    ]

    X = df[features]
    y = df[target]

    cat_cols = [
        "Store ID",
        "Product ID",
        "Category",
        "Region",
        "Weather Condition",
        "Seasonality",
    ]
    num_cols = [c for c in features if c not in cat_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", "passthrough", num_cols),
        ]
    )

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
    )
    pipeline = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, preds)

    if mlflow is not None:
        with mlflow.start_run():
            mlflow.log_metric("mae", mae)
            mlflow.sklearn.log_model(pipeline, "model")

    save_best_model(pipeline, "mae", float(mae))
    logger.info("Trained model with MAE %s", mae)
    if return_data:
        return pipeline, mae, X_test, y_test
    return pipeline, mae

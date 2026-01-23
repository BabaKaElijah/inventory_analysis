from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from logger import get_logger


logger = get_logger(__name__)


def _safe_mape(y_true: pd.Series, y_pred: pd.Series) -> float:
    mask = y_true != 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def _group_metrics(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    grouped = df.groupby(group_cols, dropna=False)
    out = grouped.apply(
        lambda g: pd.Series(
            {
                "mae": mean_absolute_error(g["y_true"], g["y_pred"]),
                "rmse": float(
                    np.sqrt(mean_squared_error(g["y_true"], g["y_pred"]))
                ),
                "mape": _safe_mape(g["y_true"], g["y_pred"]),
                "count": len(g),
            }
        )
    )
    return out.reset_index()


def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_dir: Path | str = "artifacts/reports",
) -> dict:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    y_pred = model.predict(X_test)
    overall = {
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "mape": _safe_mape(y_test, pd.Series(y_pred, index=y_test.index)),
        "count": int(len(y_test)),
    }

    eval_df = X_test.copy()
    eval_df["y_true"] = y_test.values
    eval_df["y_pred"] = y_pred

    by_store = _group_metrics(eval_df, ["Store ID"])
    by_product = _group_metrics(eval_df, ["Product ID"])
    by_store_product = _group_metrics(eval_df, ["Store ID", "Product ID"])
    by_category = _group_metrics(eval_df, ["Category"])
    by_region = _group_metrics(eval_df, ["Region"])

    (output_path / "evaluation_summary.json").write_text(
        json.dumps(overall, indent=2),
        encoding="utf-8",
    )
    by_store.to_csv(output_path / "metrics_by_store.csv", index=False)
    by_product.to_csv(output_path / "metrics_by_product.csv", index=False)
    by_store_product.to_csv(output_path / "metrics_by_store_product.csv", index=False)
    by_category.to_csv(output_path / "metrics_by_category.csv", index=False)
    by_region.to_csv(output_path / "metrics_by_region.csv", index=False)

    logger.info("Saved evaluation report to %s", output_path)
    return overall

from pathlib import Path

import pandas as pd


def test_data_schema_and_missing_values():
    data_path = Path("data") / "retail_store_inventory.csv"
    df = pd.read_csv(data_path)

    expected_columns = {
        "Date",
        "Store ID",
        "Product ID",
        "Category",
        "Region",
        "Inventory Level",
        "Units Sold",
        "Units Ordered",
        "Demand Forecast",
        "Price",
        "Discount",
        "Weather Condition",
        "Holiday/Promotion",
        "Competitor Pricing",
        "Seasonality",
    }

    assert expected_columns.issubset(set(df.columns))
    assert df.isna().sum().sum() == 0

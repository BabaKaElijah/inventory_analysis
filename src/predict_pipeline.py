import pickle
from pathlib import Path
import pandas as pd


def load_model():
    model_path = Path("artifacts") / "models" / "best_model.pkl"
    with model_path.open("rb") as f:
        return pickle.load(f)


def predict(input_df: pd.DataFrame):
    model = load_model()
    return model.predict(input_df)

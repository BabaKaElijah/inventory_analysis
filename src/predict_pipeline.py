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


def explain(input_df: pd.DataFrame, top_n: int = 10):
    model = load_model()
    preprocessor = model.named_steps["preprocess"]
    estimator = model.named_steps["model"]

    X = preprocessor.transform(input_df)
    feature_names = preprocessor.get_feature_names_out()

    import shap

    explainer = shap.TreeExplainer(estimator)
    shap_values = explainer.shap_values(X)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    if hasattr(X, "toarray"):
        row_values = X[0].toarray().ravel()
    else:
        row_values = X[0]

    contributions = []
    for name, value, impact in zip(feature_names, row_values, shap_values[0]):
        contributions.append(
            {
                "feature": str(name),
                "value": float(value),
                "impact": float(impact),
            }
        )

    contributions.sort(key=lambda x: abs(x["impact"]), reverse=True)
    return contributions[:top_n]

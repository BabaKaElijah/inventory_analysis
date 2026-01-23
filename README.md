# Inventory Demand Forecasting

End-to-end inventory analytics project that predicts daily units sold and provides explainability, evaluation reports, and production-ready serving via API and Streamlit.

## Why This Project
- Forecasts demand for store-product pairs to reduce stockouts and overstock
- Tracks experiments and saves the best model for deployment
- Provides SHAP explanations for transparent predictions
- Includes automated tests, CI workflow, and Docker support

## Highlights
- Pipeline: ingestion → transformation → training → evaluation
- Evaluation reports by store, product, category, and region
- FastAPI prediction and explanation endpoints
- Streamlit UI for interactive predictions
- MLflow tracking and artifacts management

## Tech Stack
- Python, pandas, scikit-learn
- FastAPI, Streamlit
- MLflow
- Pytest, GitHub Actions
- Docker, Docker Compose

## Project Structure
```
.
├── artifacts/
├── data/
├── src/
│   ├── app.py
│   ├── data_ingestion.py
│   ├── data_transformation.py
│   ├── model_evaluation.py
│   ├── model_pipeline.py
│   ├── model_trainer.py
│   ├── predict_pipeline.py
│   └── streamlit_app.py
├── templates/
├── tests/
├── docker-compose.yml
├── Dockerfile
└── requirements.txt
```

## Quickstart (Local)
1) Create and activate a virtual environment
2) Install dependencies
3) Train and evaluate

```cmd
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install -r requirements.txt
python src\model_pipeline.py
```

## Run FastAPI
```cmd
python -m uvicorn app:app --reload --app-dir src
```

Endpoints:
- `GET /health`
- `POST /predict`
- `POST /explain`

Docs: `http://127.0.0.1:8000/docs`

## Run Streamlit
```cmd
python -m streamlit run src/streamlit_app.py
```

App: `http://localhost:8501/`

## Docker
```cmd
docker compose up --build
```

Ensure `artifacts/models/best_model.pkl` exists on the host before running.

API: `http://127.0.0.1:8000`  
Streamlit: `http://localhost:8501`

## Tests
```cmd
python -m pytest
```

## MLflow
Runs are stored in `mlruns/`.  
Launch UI:
```cmd
python -m mlflow ui
```
Open: `http://127.0.0.1:5000`

## Evaluation Reports
Generated in `artifacts/reports/`:
- `evaluation_summary.json`
- `metrics_by_store.csv`
- `metrics_by_product.csv`
- `metrics_by_store_product.csv`
- `metrics_by_category.csv`
- `metrics_by_region.csv`

## Screenshots
Add screenshots here after running the app:
- `docs/screenshots/streamlit_home.png`
- `docs/screenshots/prediction_result.png`
- `docs/screenshots/mlflow_runs.png`

## Notes
- `Demand Forecast` is included as a feature and is highly predictive in this dataset.
- If you want a more realistic model, remove it from features and retrain.

## Author
BabaKaElijah

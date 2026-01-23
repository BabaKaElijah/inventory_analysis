# Inventory Analysis

End-to-end demand forecasting pipeline (dataset: retail_store_inventory.csv).

## Docker

Build and run API + Streamlit:

```bash
docker compose up --build
```

Ensure `artifacts/models/best_model.pkl` exists on the host before running.

API: http://127.0.0.1:8000  
Streamlit: http://127.0.0.1:8501

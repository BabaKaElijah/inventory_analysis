from fastapi import FastAPI, HTTPException, Request
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import pandas as pd

from logger import get_logger
from predict_pipeline import predict


app = FastAPI(title="Inventory Analysis API")
logger = get_logger(__name__)
templates = Jinja2Templates(directory="templates")


class PredictRequest(BaseModel):
    Store_ID: str
    Product_ID: str
    Category: str
    Region: str
    Inventory_Level: int
    Units_Ordered: int
    Demand_Forecast: float
    Price: float
    Discount: int
    Weather_Condition: str
    Holiday_Promotion: int
    Competitor_Pricing: float
    Seasonality: str
    day_of_week: int
    month: int
    day: int
    is_weekend: int

    def to_feature_dict(self) -> dict:
        return {
            "Store ID": self.Store_ID,
            "Product ID": self.Product_ID,
            "Category": self.Category,
            "Region": self.Region,
            "Inventory Level": self.Inventory_Level,
            "Units Ordered": self.Units_Ordered,
            "Demand Forecast": self.Demand_Forecast,
            "Price": self.Price,
            "Discount": self.Discount,
            "Weather Condition": self.Weather_Condition,
            "Holiday/Promotion": self.Holiday_Promotion,
            "Competitor Pricing": self.Competitor_Pricing,
            "Seasonality": self.Seasonality,
            "day_of_week": self.day_of_week,
            "month": self.month,
            "day": self.day,
            "is_weekend": self.is_weekend,
        }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/")
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict")
def predict_units_sold(payload: PredictRequest):
    try:
        features = payload.to_feature_dict()
        df = pd.DataFrame([features])
        preds = predict(df)
        prediction = float(preds[0])
        return {"prediction": prediction}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Model not found. Train the model first.")
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(exc))

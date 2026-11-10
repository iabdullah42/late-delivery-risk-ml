"""A small web API: send an order, get back the chance it ships late and whether to flag it."""
from datetime import datetime
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.data import PROMISED_DAYS, add_time_features
from src.train import MODEL_PATH

bundle = joblib.load(MODEL_PATH)
app = FastAPI(title="Late Delivery Risk API")


class Order(BaseModel):
    order_datetime: datetime
    shipping_mode: Literal["Standard Class", "Second Class", "First Class", "Same Day"]
    market: str
    region: str
    country: str
    segment: str
    payment_type: str
    main_department: str
    n_items: int = Field(ge=1)
    n_departments: int = Field(ge=1)
    total_qty: int = Field(ge=1)
    order_value: float = Field(gt=0)
    avg_discount_rate: float = Field(ge=0, le=1)


@app.get("/health")
def health():
    return {"status": "ok", "trained_through": bundle["trained_through"]}


@app.post("/predict")
def predict(order: Order):
    row = pd.DataFrame([order.model_dump()])
    row["order_date"] = row.pop("order_datetime")
    row["scheduled_days"] = row["shipping_mode"].map(PROMISED_DAYS)
    row = add_time_features(row)
    prob = float(bundle["model"].predict_proba(row[bundle["features"]])[0, 1])
    return {"late_probability": round(prob, 3),
            "flag": prob >= bundle["threshold"],
            "threshold": bundle["threshold"]}
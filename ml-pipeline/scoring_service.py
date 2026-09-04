"""
Live scoring microservice for Merchant Risk Shield.

Loads the two models trained by train_models.py (real XGBoost + Random Forest,
saved via joblib) and exposes them over HTTP so the frontend can score a
brand-new order or dispute in real time, instead of only showing pre-computed
demo rows.

Run:
    pip install fastapi uvicorn joblib scikit-learn xgboost pandas numpy
    uvicorn scoring_service:app --reload --port 8000

Then:
    POST http://localhost:8000/score-order
    POST http://localhost:8000/score-dispute
"""

import json
import os

import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

HERE = os.path.dirname(os.path.abspath(__file__))

app = FastAPI(title="Merchant Risk Shield — Scoring Service")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # fine for a hackathon demo; lock this down for prod
    allow_methods=["*"],
    allow_headers=["*"],
)

model1 = joblib.load(os.path.join(HERE, "return_risk_model.pkl"))
with open(os.path.join(HERE, "return_risk_columns.json")) as f:
    COLS1 = json.load(f)

model2 = joblib.load(os.path.join(HERE, "chargeback_win_model.pkl"))
with open(os.path.join(HERE, "chargeback_win_columns.json")) as f:
    COLS2 = json.load(f)

CATEGORIES = ["Electronics", "Fashion", "Fitness", "Home & Kitchen", "Beauty", "Mobiles"]
REASONS = ["Item not received", "Item not as described", "Transaction not recognised", "Duplicate charge", "Defective item"]


class OrderInput(BaseModel):
    order_value: float = Field(..., example=8500)
    is_cod: bool = True
    past_orders: int = Field(..., example=12)
    hist_return_rate: float = Field(..., ge=0, le=1, example=0.4)
    days_since_signup: int = Field(..., example=90)
    prior_disputes: int = 0
    device_ip_mismatch: bool = False
    night_order: bool = False
    coupon_used: bool = False
    delivery_pincode_risk: float = Field(..., ge=0, le=1, example=0.3)
    category: str = Field(..., example="Electronics")


class DisputeInput(BaseModel):
    order_value: float = Field(..., example=6500)
    has_delivery_proof: bool = True
    has_signed_pod: bool = True
    ip_match: bool = True
    tracking_valid: bool = True
    comm_logs: int = 1
    days_to_file: int = Field(..., example=5)
    prior_customer_disputes: int = 0
    reason: str = Field(..., example="Item not received")


def _row_to_vector(row: dict, columns: list, onehot_prefix: str, onehot_value: str):
    vec = {c: 0 for c in columns}
    for k, v in row.items():
        if k in vec:
            vec[k] = int(v) if isinstance(v, bool) else v
    onehot_col = f"{onehot_prefix}_{onehot_value}"
    if onehot_col in vec:
        vec[onehot_col] = 1
    return pd.DataFrame([vec], columns=columns)


@app.get("/health")
def health():
    return {"ok": True, "model1_features": len(COLS1), "model2_features": len(COLS2)}


@app.post("/score-order")
def score_order(order: OrderInput):
    if order.category not in CATEGORIES:
        return {"error": f"category must be one of {CATEGORIES}"}
    row = order.dict()
    row.pop("category")
    X = _row_to_vector(row, COLS1, "category", order.category)
    proba = float(model1.predict_proba(X)[0, 1])
    risk_score = round(proba * 100)
    band = "High" if risk_score >= 65 else "Medium" if risk_score >= 35 else "Low"
    return {"risk_score": risk_score, "band": band, "model": "XGBoost (return_risk_model.pkl)"}


@app.post("/score-dispute")
def score_dispute(dispute: DisputeInput):
    if dispute.reason not in REASONS:
        return {"error": f"reason must be one of {REASONS}"}
    row = dispute.dict()
    row.pop("reason")
    X = _row_to_vector(row, COLS2, "reason", dispute.reason)
    proba = float(model2.predict_proba(X)[0, 1])
    win_probability = round(proba * 100)
    return {"win_probability": win_probability, "model": "Random Forest (chargeback_win_model.pkl)"}

"""
Trains:
  Model 1 — Return Risk Scorer        (XGBoost classifier)
  Model 2 — Chargeback Win Predictor  (Random Forest classifier)

Outputs real precision/recall/F1/confusion-matrix/feature-importance,
and dumps per-row predictions the frontend can render.
"""

import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from xgboost import XGBClassifier

import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = HERE

# ---------------------------------------------------------------------------
# Model 1 — Return Risk Scorer (XGBoost)
# ---------------------------------------------------------------------------

orders = pd.read_csv(f"{OUT}/orders.csv")

feat_cols_1 = [
    "order_value", "is_cod", "past_orders", "hist_return_rate", "days_since_signup",
    "prior_disputes", "device_ip_mismatch", "night_order", "coupon_used", "delivery_pincode_risk",
]
cat_cols_1 = ["category"]
X1 = pd.get_dummies(orders[feat_cols_1 + cat_cols_1], columns=cat_cols_1)
y1 = orders["returned"]

X1_train, X1_test, y1_train, y1_test, idx_train, idx_test = train_test_split(
    X1, y1, orders.index, test_size=0.25, random_state=42, stratify=y1
)

model1 = XGBClassifier(
    n_estimators=250, max_depth=4, learning_rate=0.08,
    subsample=0.85, colsample_bytree=0.85, eval_metric="logloss", random_state=42,
)
model1.fit(X1_train, y1_train)

proba1_test = model1.predict_proba(X1_test)[:, 1]
pred1_test = (proba1_test >= 0.5).astype(int)

metrics1 = dict(
    precision=round(precision_score(y1_test, pred1_test) * 100, 1),
    recall=round(recall_score(y1_test, pred1_test) * 100, 1),
    f1=round(f1_score(y1_test, pred1_test) * 100, 1),
    auc=round(roc_auc_score(y1_test, proba1_test) * 100, 1),
)
cm1 = confusion_matrix(y1_test, pred1_test).tolist()
fi1 = sorted(zip(X1.columns, model1.feature_importances_), key=lambda t: -t[1])[:8]
fi1 = [{"feature": f, "importance": round(float(v), 4)} for f, v in fi1]

print("Model 1 (Return Risk / XGBoost):", metrics1)

joblib.dump(model1, f"{OUT}/return_risk_model.pkl")
with open(f"{OUT}/return_risk_columns.json", "w") as f:
    json.dump(list(X1.columns), f)

# score ALL orders (0-100 risk score) for the demo dataset
orders["risk_score"] = (model1.predict_proba(X1)[:, 1] * 100).round().astype(int)

# ---------------------------------------------------------------------------
# Model 2 — Chargeback Win Predictor (Random Forest)
# ---------------------------------------------------------------------------

disputes = pd.read_csv(f"{OUT}/disputes.csv")

feat_cols_2 = [
    "order_value", "has_delivery_proof", "has_signed_pod", "ip_match", "tracking_valid",
    "comm_logs", "days_to_file", "prior_customer_disputes",
]
cat_cols_2 = ["reason"]
X2 = pd.get_dummies(disputes[feat_cols_2 + cat_cols_2], columns=cat_cols_2)
y2 = disputes["won"]

X2_train, X2_test, y2_train, y2_test = train_test_split(
    X2, y2, test_size=0.25, random_state=42, stratify=y2
)

model2 = RandomForestClassifier(
    n_estimators=300, max_depth=7, min_samples_leaf=4, random_state=42, class_weight="balanced_subsample"
)
model2.fit(X2_train, y2_train)

proba2_test = model2.predict_proba(X2_test)[:, 1]
pred2_test = (proba2_test >= 0.5).astype(int)

metrics2 = dict(
    precision=round(precision_score(y2_test, pred2_test) * 100, 1),
    recall=round(recall_score(y2_test, pred2_test) * 100, 1),
    f1=round(f1_score(y2_test, pred2_test) * 100, 1),
    auc=round(roc_auc_score(y2_test, proba2_test) * 100, 1),
)
cm2 = confusion_matrix(y2_test, pred2_test).tolist()
fi2 = sorted(zip(X2.columns, model2.feature_importances_), key=lambda t: -t[1])[:8]
fi2 = [{"feature": f, "importance": round(float(v), 4)} for f, v in fi2]

print("Model 2 (Chargeback Win / Random Forest):", metrics2)

joblib.dump(model2, f"{OUT}/chargeback_win_model.pkl")
with open(f"{OUT}/chargeback_win_columns.json", "w") as f:
    json.dump(list(X2.columns), f)

disputes["win_probability"] = (model2.predict_proba(X2)[:, 1] * 100).round().astype(int)

# ---------------------------------------------------------------------------
# Export a demo-sized slice + metrics for the frontend
# ---------------------------------------------------------------------------

FIRST_NAMES_DEMO = None  # keep Faker-generated names already in df

demo_orders = orders.sample(n=40, random_state=7).sort_values("risk_score", ascending=False)
demo_orders_out = demo_orders[[
    "order_id", "customer", "category", "order_value", "is_cod", "past_orders",
    "hist_return_rate", "risk_score",
]].to_dict(orient="records")

demo_disputes = disputes.sample(n=18, random_state=7).sort_values("win_probability", ascending=False)
demo_disputes_out = demo_disputes[[
    "dispute_id", "order_id", "reason", "order_value", "has_signed_pod", "ip_match",
    "tracking_valid", "comm_logs", "win_probability",
]].to_dict(orient="records")

# monthly saved-estimate trend derived from real risk scores (illustrative rollup)
rng = np.random.default_rng(3)
months = ["Mar", "Apr", "May", "Jun", "Jul", "Aug"]
base = orders["risk_score"].mean() / 100 * orders["order_value"].mean()
trend = [round(base * (0.55 + 0.09 * i) * 6000 / 1e6 * (1 + rng.normal(0, 0.05)), 2) for i in range(6)]

export = {
    "generated_at": "2026-09-02",
    "dataset_sizes": {"orders": len(orders), "disputes": len(disputes)},
    "model1": {
        "name": "Return Risk Scorer",
        "algorithm": "XGBoost (250 trees, depth 4)",
        "metrics": metrics1,
        "confusion_matrix": cm1,
        "feature_importance": fi1,
    },
    "model2": {
        "name": "Chargeback Win Predictor",
        "algorithm": "Random Forest (300 trees, depth 7, balanced)",
        "metrics": metrics2,
        "confusion_matrix": cm2,
        "feature_importance": fi2,
    },
    "orders": demo_orders_out,
    "disputes": demo_disputes_out,
    "saved_trend": [{"month": m, "saved_inr_k": v} for m, v in zip(months, trend)],
}

with open(f"{OUT}/export.json", "w") as f:
    json.dump(export, f, indent=2, default=str)

print("\nExported ->", f"{OUT}/export.json")
print("Demo orders:", len(demo_orders_out), " Demo disputes:", len(demo_disputes_out))

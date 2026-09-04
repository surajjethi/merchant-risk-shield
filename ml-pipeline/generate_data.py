
"""
Merchant Risk Shield — synthetic data generator
Creates two datasets:
  1. orders.csv      -> used to train the Return-Risk Scorer
  2. disputes.csv    -> used to train the Chargeback Win Predictor

Patterns are hand-tuned to be learnable (not random noise) so the models
below produce honest, non-trivial precision/recall numbers.
"""

import numpy as np
import pandas as pd
from faker import Faker
import random
import os


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = HERE

rng = np.random.default_rng(42)
random.seed(42)

fake = Faker("en_IN")
Faker.seed(42)

N_ORDERS = 6000
N_DISPUTES = 1400

CATEGORIES = [
    "Electronics",
    "Fashion",
    "Fitness",
    "Home & Kitchen",
    "Beauty",
    "Mobiles",
]

CATEGORY_RETURN_BASE = {
    "Electronics": 0.22,
    "Fashion": 0.28,
    "Fitness": 0.10,
    "Home & Kitchen": 0.14,
    "Beauty": 0.08,
    "Mobiles": 0.30,
}


# ---------------------------------------------------------------------------
# 1. ORDERS — Return Risk Scorer
# ---------------------------------------------------------------------------

rows = []

for i in range(N_ORDERS):

    category = random.choices(
        CATEGORIES,
        weights=[0.24, 0.22, 0.10, 0.16, 0.10, 0.18]
    )[0]

    past_orders = int(rng.gamma(3.5, 4))
    past_orders = max(1, past_orders)

    base_return_rate = CATEGORY_RETURN_BASE[category]

    # Customer-level "return proneness" latent trait
    proneness = np.clip(
        rng.beta(1.6, 5.5),
        0,
        1
    )

    hist_return_rate = np.clip(
        base_return_rate * 0.4
        + proneness * 0.9
        + rng.normal(0, 0.05),
        0,
        1
    )

    order_value = float(
        np.clip(
            rng.lognormal(mean=8.4, sigma=0.65),
            200,
            60000
        )
    )

    is_cod = random.random() < (
        0.35 + proneness * 0.4
    )

    days_since_signup = int(
        rng.exponential(220)
    )

    prior_disputes = np.random.poisson(
        proneness * 1.2
    )

    device_ip_mismatch = int(
        random.random() < (
            0.05 + proneness * 0.25
        )
    )

    night_order = int(
        random.random() < 0.15
    )

    coupon_used = int(
        random.random() < 0.3
    )

    delivery_pincode_risk = np.clip(
        rng.beta(2, 6)
        + proneness * 0.3,
        0,
        1
    )

    # -----------------------------------------------------------------------
    # Ground truth label:
    # Did this order actually get returned/refund-abused?
    # -----------------------------------------------------------------------

    logit = (
        -2.6
        + 4.2 * proneness
        + 2.6 * hist_return_rate
        + 1.1 * device_ip_mismatch
        + 0.7 * is_cod
        + 0.6 * delivery_pincode_risk
        + 0.4 * prior_disputes
        + 0.3 * night_order
        - 0.15 * (days_since_signup > 365)
        + rng.normal(0, 0.35)
    )

    prob_return = 1 / (
        1 + np.exp(-logit)
    )

    returned = int(
        random.random() < prob_return
    )

    rows.append(
        dict(
            order_id=f"ORD-{10000 + i}",
            customer=fake.name(),
            category=category,
            order_value=round(order_value, 2),
            is_cod=int(is_cod),
            past_orders=past_orders,
            hist_return_rate=round(hist_return_rate, 3),
            days_since_signup=days_since_signup,
            prior_disputes=int(prior_disputes),
            device_ip_mismatch=device_ip_mismatch,
            night_order=night_order,
            coupon_used=coupon_used,
            delivery_pincode_risk=round(
                delivery_pincode_risk,
                3
            ),
            returned=returned,
        )
    )


orders_df = pd.DataFrame(rows)

orders_df.to_csv(
    os.path.join(OUT, "orders.csv"),
    index=False
)

print(
    "orders.csv ->",
    orders_df.shape,
    " positive rate:",
    orders_df.returned.mean().round(3)
)


# ---------------------------------------------------------------------------
# 2. DISPUTES — Chargeback Win Predictor
# ---------------------------------------------------------------------------

REASONS = [
    "Item not received",
    "Item not as described",
    "Transaction not recognised",
    "Duplicate charge",
    "Defective item",
]

rows = []

for i in range(N_DISPUTES):

    reason = random.choice(REASONS)

    has_delivery_proof = int(
        random.random() < 0.72
    )

    has_signed_pod = int(
        has_delivery_proof
        and random.random() < 0.8
    )

    ip_match = int(
        random.random() < 0.6
    )

    tracking_valid = int(
        has_delivery_proof
        and random.random() < 0.9
    )

    comm_logs = np.random.poisson(1.4)

    order_value = float(
        np.clip(
            rng.lognormal(mean=8.5, sigma=0.6),
            300,
            50000
        )
    )

    days_to_file = int(
        rng.exponential(10)
    )

    prior_customer_disputes = np.random.poisson(
        0.4
    )

    reason_strength = {
        "Item not received": 0.0,
        "Item not as described": -0.3,
        "Transaction not recognised": 0.2,
        "Duplicate charge": 0.6,
        "Defective item": -0.5,
    }[reason]

    logit = (
        -1.5
        + 2.2 * has_signed_pod
        + 1.3 * tracking_valid
        + 1.0 * ip_match
        + 0.28 * np.minimum(comm_logs, 5)
        + reason_strength
        - 0.5 * prior_customer_disputes
        - 0.02 * np.minimum(days_to_file, 30)
        + rng.normal(0, 0.4)
    )

    prob_win = 1 / (
        1 + np.exp(-logit)
    )

    won = int(
        random.random() < prob_win
    )

    rows.append(
        dict(
            dispute_id=f"DSP-{2000 + i}",
            order_id=(
                f"ORD-{10000 + random.randint(0, N_ORDERS - 1)}"
            ),
            reason=reason,
            order_value=round(order_value, 2),
            has_delivery_proof=has_delivery_proof,
            has_signed_pod=has_signed_pod,
            ip_match=ip_match,
            tracking_valid=tracking_valid,
            comm_logs=int(comm_logs),
            days_to_file=days_to_file,
            prior_customer_disputes=int(
                prior_customer_disputes
            ),
            won=won,
        )
    )


disputes_df = pd.DataFrame(rows)

disputes_df.to_csv(
    os.path.join(OUT, "disputes.csv"),
    index=False
)

print(
    "disputes.csv ->",
    disputes_df.shape,
    " win rate:",
    disputes_df.won.mean().round(3)
)


# Merchant Risk Shield

A practical risk and dispute management dashboard for merchants.

Merchant Risk Shield brings two common merchant problems into one place: identifying orders that are more likely to be returned, and deciding which chargebacks are worth contesting.

The dashboard combines machine-learning predictions with the evidence already available for each case, so the output is not just a score — it leads to a suggested action.

##  Live Demo

**[Open Merchant Risk Shield](https://merchant-risk-shield-frontend.onrender.com)**

**Demo account**
- Email: `demo@merchant.com`
- Password: `demo123`

> The project uses synthetic data for the demo. No real customer or payment data is used.

---

## What the project does

### 1. Order Return Risk

The Orders section uses an **XGBoost** model to estimate the likelihood that an order will be returned.

A merchant can also enter a new order in the live scoring panel and get a prediction from the trained model.

### 2. Chargeback Win Prediction

The Disputes section uses a **Random Forest** model to estimate the probability that a merchant can successfully win a chargeback dispute.

The prediction is used to group cases into practical decision bands:

- **Below 40%** → Do not contest
- **40–69%** → Manual review
- **70% and above** → Contest

These are decision-support signals, not guarantees.

### 3. Evidence Desk

For a dispute, the system looks at the evidence actually available in the case, including:

- Signed proof of delivery
- Courier tracking validity
- Device/IP consistency
- Previous support communication

It then generates an evidence letter based on the case details and predicted outcome.

### 4. Merchant Actions

The dashboard also supports simple operational actions such as:

- Verify an order
- Block COD for an order
- Generate an evidence letter
- Submit a dispute

---

## How it works

```text
                    Merchant Risk Shield
                           │
             ┌─────────────┴─────────────┐
             │                           │
             ▼                           ▼
       Order Risk                  Dispute Risk
             │                           │
          XGBoost                   Random Forest
             │                           │
             ▼                           ▼
       Return Risk             Chargeback Win Probability
             │                           │
             └─────────────┬─────────────┘
                           ▼
                    Merchant Dashboard
                           │
             ┌─────────────┴─────────────┐
             ▼                           ▼
       Merchant Actions            Evidence Desk
       Verify / Block          Generate / Submit Letter

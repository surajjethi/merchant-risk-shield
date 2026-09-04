Merchant Risk Shield

A practical AI-powered risk and dispute management dashboard for
merchants.

Merchant Risk Shield brings two common merchant problems into one place:
identifying orders that are more likely to be returned and deciding
which chargebacks are worth contesting.

Instead of showing machine-learning scores in isolation, the system
combines predictions with the evidence already available for each case
and turns them into actionable merchant decisions.

Live Demo

Frontend:
https://merchant-risk-shield-frontend.onrender.com

Demo account

Email: demo@merchant.com
Password: demo123

The Problem

Merchants deal with two recurring problems that can directly affect
revenue:

Product returns --- some orders have a higher likelihood of
being returned, but identifying them early can be difficult.

Chargebacks --- when a customer disputes a transaction,
merchants need to decide whether the case is strong enough to
contest.

Handling these cases manually means going through large amounts of
order, tracking, payment, delivery and communication information.

Merchant Risk Shield provides a single workflow to help merchants:

identify potentially high-risk orders

estimate return risk

predict the likelihood of winning a chargeback

review supporting evidence

generate evidence-based dispute letters

take merchant actions such as verification or blocking

prioritize cases that are more likely to result in recovery

What Merchant Risk Shield Does

The platform has two main ML-driven workflows.

1. Order Return Risk

The system uses an XGBoost model to estimate the likelihood that an
order will be returned.

The prediction is displayed directly inside the Orders dashboard so
merchants can quickly focus on orders that may need attention.

The model uses order-level and behavioural signals such as:

order value

customer behaviour

product information

delivery-related information

payment characteristics

historical signals

The goal is not to automatically reject an order, but to provide an
early risk signal that can support merchant decision-making.

2. Chargeback Win Prediction

The second workflow focuses on disputes.

A Random Forest classifier estimates the probability that a merchant
can successfully win a chargeback.

The result is converted into practical decision bands:

Win Probability   Suggested Action

Below 40%         Do not contest
40--69%           Manual review
70% and above     Contest

These thresholds are decision-support signals for the prototype, not
guarantees of an actual chargeback outcome.

Evidence Desk

A prediction becomes much more useful when the merchant can understand
whether there is enough evidence to support the case.

For eligible disputes, Merchant Risk Shield generates an evidence-based
dispute letter using information available for that specific dispute.

The evidence can include:

signed proof of delivery

valid tracking

IP/address match

customer communication history

order value

dispute reason

predicted win probability

The generated letter is tied to the actual dispute instead of relying on
one generic template.

Evidence workflow

Dispute
   ↓
ML Win Probability
   ↓
Evidence Review
   ↓
Decision
   ↓
Generate Evidence Letter
   ↓
Submit Dispute

Merchant Actions

The dashboard goes beyond analytics by connecting predictions to
operational actions.

For orders, merchants can:

verify an order

block an order

For disputes, merchants can:

review the predicted win probability

generate an evidence letter

submit the dispute

This makes the system closer to a merchant operations tool rather than
just an ML demonstration.

Dashboard

Merchant Login



Order Risk Monitoring



Chargeback & Dispute Management



Model Lab



Architecture



The application is split into three services:

                         Merchant
                            │
                            ▼
                 ┌─────────────────────┐
                 │      Frontend       │
                 │     React / Vite    │
                 │      Render         │
                 └──────────┬──────────┘
                            │
                 ┌──────────┴──────────┐
                 │                     │
                 ▼                     ▼
       ┌──────────────────┐   ┌──────────────────┐
       │ Backend API      │   │ ML Scoring API   │
       │ Node / Express   │   │ Python / FastAPI │
       │ Render           │   │ Render           │
       └────────┬─────────┘   └────────┬─────────┘
                │                      │
                ▼                      ▼
        Orders / Disputes       XGBoost + RF Models
        Actions / Evidence      Live Predictions

Service Responsibilities

Frontend

React + Vite dashboard

merchant login

order risk monitoring

dispute management

model metrics

live scoring panels

merchant actions

Backend

order and dispute APIs

merchant actions

evidence letter generation

dispute submission

serving demo/exported data

ML Pipeline

synthetic dataset generation

model training

model serialization

FastAPI scoring service

live order and dispute predictions

Machine Learning

Merchant Risk Shield currently uses two classification models.

Return Risk --- XGBoost

The XGBoost model predicts whether an order is likely to be returned.

Metric        Score

Precision     68.1%
Recall        58.4%
F1 Score      62.9%
AUC           75.0%

Chargeback Win --- Random Forest

The Random Forest model predicts the likelihood of successfully winning
a chargeback.

Metric        Score

Precision     86.4%
Recall        81.5%
F1 Score      83.9%
AUC           85.4%

These results come from a held-out test split of the synthetic dataset.
They are included to demonstrate the current prototype's model
performance and should not be interpreted as production accuracy.

Model Lab

The Model Lab provides visibility into how the models are performing
instead of treating them as a black box.

It currently displays:

Precision

Recall

F1 Score

AUC

confusion matrices

feature importance

This gives the merchant and evaluator a quick view of both model quality
and the signals influencing predictions.

Dataset

The current prototype uses synthetic merchant data.

6,000 orders
1,400 disputes

Synthetic data is used so the project can demonstrate the complete
workflow without exposing real customer or payment information.

The data generation pipeline is included in the repository and can be
regenerated locally.

Tech Stack

Frontend

React

Vite

JavaScript

CSS

Backend

Node.js

Express.js

Machine Learning

Python

FastAPI

XGBoost

Scikit-learn

Random Forest

Pandas

NumPy

Joblib

Faker

Deployment

Render

Project Structure

merchant-risk-shield/
│
├── frontend/
│   ├── src/
│   │   ├── App.jsx
│   │   └── main.jsx
│   ├── index.html
│   ├── package.json
│   └── vite.config.js
│
├── backend/
│   ├── data/
│   │   └── export.json
│   ├── server.js
│   ├── package.json
│   └── package-lock.json
│
├── ml-pipeline/
│   ├── generate_data.py
│   ├── train_models.py
│   ├── scoring_service.py
│   ├── orders.csv
│   ├── disputes.csv
│   ├── return_risk_model.pkl
│   ├── chargeback_win_model.pkl
│   ├── return_risk_columns.json
│   ├── chargeback_win_columns.json
│   └── requirements.txt
│
├── docs/
│   ├── architecture.png
│   └── screenshots/
│       ├── login.png
│       ├── orders.png
│       ├── disputes.png
│       └── model-lab.png
│
├── README.md
└── .gitignore

Running Locally

1. Clone the repository

git clone https://github.com/surajjethi/merchant-risk-shield.git
cd merchant-risk-shield

2. Start the Backend

Open a terminal:

cd backend
npm install
npm start

The backend runs on:

http://localhost:4000

Health check:

http://localhost:4000/api/health

3. Start the ML Scoring Service

Open another terminal:

cd ml-pipeline

Create a virtual environment if needed:

python -m venv venv

Activate it on Windows:

venv\Scripts\activate

Install dependencies:

pip install -r requirements.txt

Start FastAPI:

uvicorn scoring_service:app --host 0.0.0.0 --port 8000

The ML service runs on:

http://localhost:8000

Health check:

http://localhost:8000/health

4. Start the Frontend

Open another terminal:

cd frontend
npm install
npm run dev

Open:

http://localhost:5173

Use the demo credentials shown above to enter the dashboard.

Local Architecture

When running locally, the services communicate like this:

React Frontend
localhost:5173
      │
      ├──────────────► Node / Express
      │                localhost:4000
      │
      └──────────────► FastAPI ML Service
                       localhost:8000

The frontend communicates directly with the ML scoring service for live
predictions.

The backend handles merchant actions, dispute workflows and
evidence-letter generation.

Regenerating the Dataset

The ML pipeline includes a synthetic data generator.

From the ml-pipeline directory:

python generate_data.py

This regenerates the order and dispute datasets.

Models can then be retrained using:

python train_models.py

The training pipeline exports the trained models and model information
used by the dashboard.

Deployment

The current prototype is deployed using Render.

Frontend

https://merchant-risk-shield-frontend.onrender.com

Backend API

https://merchant-risk-shield-api.onrender.com

ML Scoring Service

https://merchant-risk-shield.onrender.com

The frontend is configured to communicate with the deployed backend and
ML scoring service.

Why This Approach?

A merchant usually does not need another dashboard full of isolated
predictions.

What they need is a workflow:

Identify Risk
      ↓
Understand the Case
      ↓
Check Supporting Evidence
      ↓
Decide What to Do
      ↓
Take Action

That is the main idea behind Merchant Risk Shield.

The ML models provide the prediction layer, while the dashboard connects
those predictions to evidence and merchant actions.

Key Takeaways

Merchant Risk Shield demonstrates how machine learning can be integrated
into an actual merchant workflow rather than being presented as a
standalone model.

The prototype combines:

Return risk prediction

Chargeback win prediction

Evidence-based dispute generation

Merchant actions

Model transparency

Live ML scoring

A deployable web dashboard

The focus is on turning a prediction into an actionable decision.

Future Improvements

A production version could be extended with:

integration with real payment and order data

real-time event streaming

merchant-specific model calibration

model monitoring and drift detection

explainable AI for individual predictions

role-based access control

production-grade authentication

automated evidence collection

direct integration with payment processor dispute APIs

feedback loops from actual dispute outcomes

model retraining based on new merchant data

Buildathon Context

Merchant Risk Shield was built as an AI-powered merchant risk and
dispute management prototype for the Razorpay AI Buildathon.

The project focuses on a practical question:

How can machine learning help merchants decide where to act first and
which disputes are worth fighting?

Rather than only predicting risk, the system connects predictions to
evidence, recommended actions and an operational workflow.

Disclaimer

This project is a prototype built using synthetic data.

The ML predictions and decision thresholds are intended for
demonstration and decision-support purposes only. They are not
guarantees of actual returns, fraud outcomes or chargeback results.

No real customer payment or personally identifiable information is used
in the prototype.

Author

Suraj Singh Jethi

GitHub:
surajjethi/merchant-risk-shield

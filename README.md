# Merchant Risk Shield

Unzip this folder anywhere and open it in VS Code (`File → Open Folder`).
Four pieces now, run each from its own terminal tab.

```
merchant-risk-shield/
├── ml-pipeline/     Python: synthetic data, model training, AND a live-scoring API
│   ├── generate_data.py        synthetic orders.csv / disputes.csv
│   ├── train_models.py         trains + saves XGBoost + Random Forest as .pkl
│   ├── scoring_service.py      FastAPI service — scores NEW orders/disputes live
│   ├── return_risk_model.pkl / chargeback_win_model.pkl   already-trained models
│   └── requirements.txt
├── backend/         Node.js / Express API serving the pre-computed demo data
└── frontend/        React (Vite) dashboard — the actual UI
```

## Fastest path to a full demo (3 terminals)

**Terminal 1 — live scoring service** (makes the "Score live" panels in Orders/Disputes work):

```bash
cd ml-pipeline
pip install -r requirements.txt
uvicorn scoring_service:app --reload --port 8000
```

Leave this running. Test it worked: open `http://localhost:8000/health` in a browser —
you should see `{"ok":true,...}`.

**Terminal 2 — frontend:**

```bash
cd frontend
npm install
npm run dev
```

Open **http://localhost:5173**. The dashboard works even without Terminal 1 running
(all the demo rows are baked in) — but the "Score a new order/dispute live" panels
at the top of Orders and Disputes need Terminal 1 to actually call the real model.

**Terminal 3 — backend** (optional, not wired to the frontend yet):

```bash
cd backend
npm install
npm start
```

## What's actually "live" here

- The **Orders** and **Disputes** tabs ship with 40/18 pre-scored demo rows (from
  `export.json`, produced by a real train/test split — not hand-picked numbers).
- The **"Score a new order live"** / **"Score a new dispute live"** panels at the
  top of each tab call `scoring_service.py` over HTTP, which loads the actual
  `.pkl` files and runs `model.predict_proba()` on whatever you type in. Type in
  a risky-looking order and watch the score change in real time — that's the
  real XGBoost model, not a canned response. Rows you add this way get a small
  "LIVE" badge.
- **Model Lab** tab shows the real precision/recall/F1/AUC/confusion-matrix/
  feature-importance from the last `train_models.py` run.

## Regenerating data / retraining

```bash
cd ml-pipeline
python3 generate_data.py     # fresh synthetic orders.csv / disputes.csv
python3 train_models.py      # retrains, re-saves the .pkl files, prints real metrics
```

Restart `scoring_service.py` after retraining so it picks up the new `.pkl` files.

## Requirements

- Node.js 18+ and npm (frontend/backend)
- Python 3.9+ and pip (ml-pipeline)

## Troubleshooting

- **"Scoring service unreachable" in the UI** → Terminal 1 isn't running, or died.
  Check it printed `Uvicorn running on http://127.0.0.1:8000`.
- **Blank page / import errors in frontend** → `npm install` didn't finish cleanly;
  delete `node_modules` and re-run it.
- **Port already in use** → change the port in `frontend/vite.config.js`
  (frontend), `PORT=4001 npm start` (backend), or `--port 8001` (scoring service).
- **pip install fails on Windows for xgboost** → upgrade pip first:
  `python -m pip install --upgrade pip`, then retry.

## Deploying for demo day (recommended if you have time)

- **Frontend** → push `frontend/` to GitHub, import into
  [vercel.com](https://vercel.com) (auto-detects Vite), deploy. ~3 minutes.
- **Scoring service** → push `ml-pipeline/` to GitHub, deploy on
  [render.com](https://render.com) as a Python web service
  (`uvicorn scoring_service:app --host 0.0.0.0 --port $PORT`). Free tier works.
- Then update `SCORING_URL` at the top of `frontend/src/App.jsx` from
  `http://localhost:8000` to your Render URL, and redeploy the frontend.

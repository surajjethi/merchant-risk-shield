/**
 * Merchant Risk Shield — API server
 * Serves the Python/XGBoost + Random Forest model output and handles the
 * two live merchant actions: order verification/COD-block, and dispute
 * evidence-letter generation/submission.
 *
 * Run:
 *   npm install
 *   npm start        (defaults to http://localhost:4000)
 */

const express = require("express");
const cors = require("cors");
const fs = require("fs");
const path = require("path");

const app = express();
app.use(cors());
app.use(express.json());

const DATA_PATH = path.join(__dirname, "data", "export.json");
const raw = JSON.parse(fs.readFileSync(DATA_PATH, "utf-8"));

// ---------------------------------------------------------------------------
// In-memory state, seeded from the model export. Resets on server restart —
// swap this for a real DB (Postgres/Mongo) to persist between deploys.
// ---------------------------------------------------------------------------

const state = {
  model1: raw.model1,
  model2: raw.model2,
  datasetSizes: raw.dataset_sizes,
  savedTrend: raw.saved_trend,
  orders: raw.orders.map((o) => ({ ...o, status: "pending" })),
  disputes: raw.disputes.map((d) => ({
    ...d,
    status: "pending",
    letter: null,
  })),
};


// ---------------------------------------------------------------------------
// Evidence Letter Generator
// ---------------------------------------------------------------------------

function buildLetter(d) {
  const probability = Number(d.win_probability) || 0;

  const amount = `₹${Number(d.order_value).toLocaleString("en-IN")}`;

  const disputeCount =
    Number(state.datasetSizes.disputes).toLocaleString("en-IN");

  // -------------------------------------------------------------------------
  // 1. Collect evidence actually available in the dataset
  // -------------------------------------------------------------------------

  const supportingEvidence = [];
  const evidenceGaps = [];

  if (d.has_signed_pod) {
    supportingEvidence.push(
      "Signed proof of delivery is available for this shipment."
    );
  } else {
    evidenceGaps.push(
      "No signed proof-of-delivery record was located."
    );
  }

  if (d.tracking_valid) {
    supportingEvidence.push(
      "Courier tracking history is available and marked valid."
    );
  } else {
    evidenceGaps.push(
      "Courier tracking history could not be fully verified."
    );
  }

  if (d.ip_match) {
    supportingEvidence.push(
      "The device/IP activity matches known customer account activity."
    );
  } else {
    evidenceGaps.push(
      "Device/IP consistency with known customer activity could not be established."
    );
  }

  if (Number(d.comm_logs) > 0) {
    supportingEvidence.push(
      `${d.comm_logs} prior support communication(s) are available in the case record.`
    );
  } else {
    evidenceGaps.push(
      "No prior support communication is available in the current case record."
    );
  }


  // -------------------------------------------------------------------------
  // 2. Determine decision band from model probability
  // -------------------------------------------------------------------------

  let decision;
  let decisionTitle;
  let decisionExplanation;

  if (probability < 40) {
    decision = "DO NOT CONTEST";

    decisionTitle = "Low predicted win probability";

    decisionExplanation =
      "The available evidence does not currently provide strong support for a successful representment. The merchant should consider accepting the chargeback or obtaining additional evidence before contesting.";
  } else if (probability < 70) {
    decision = "MANUAL REVIEW";

    decisionTitle = "Uncertain predicted outcome";

    decisionExplanation =
      "The evidence package contains both supporting signals and evidence gaps. A manual review is recommended before deciding whether to submit a representment.";
  } else {
    decision = "CONTEST";

    decisionTitle = "Strong predicted win probability";

    decisionExplanation =
      "The available evidence provides strong support for a representment. The merchant can proceed with a chargeback response using the documented evidence.";
  }


  // -------------------------------------------------------------------------
  // 3. Build evidence section
  // -------------------------------------------------------------------------

  const evidenceLines = [];

  if (supportingEvidence.length > 0) {
    evidenceLines.push("Supporting evidence:");

    supportingEvidence.forEach((item) => {
      evidenceLines.push(`  • ${item}`);
    });
  }

  if (evidenceGaps.length > 0) {
    evidenceLines.push("");

    evidenceLines.push("Evidence gaps:");

    evidenceGaps.forEach((item) => {
      evidenceLines.push(`  • ${item}`);
    });
  }


  // -------------------------------------------------------------------------
  // 4. LOW PROBABILITY
  // -------------------------------------------------------------------------

  if (probability < 40) {
    const lines = [
      `Subject: Chargeback Assessment — Dispute ${d.dispute_id} / Order ${d.order_id}`,
      ``,
      `To the Merchant Dispute Resolution Team,`,
      ``,

      `CASE DETAILS`,
      `Dispute: ${d.dispute_id}`,
      `Order: ${d.order_id}`,
      `Transaction value: ${amount}`,
      `Dispute reason: "${d.reason}"`,
      ``,

      `RISK ASSESSMENT`,
      `${decisionTitle}: ${probability}% predicted win probability.`,
      `Recommended action: ${decision}`,
      ``,

      `EVIDENCE ASSESSMENT`,
      ...evidenceLines,
      ``,

      `RECOMMENDATION`,
      decisionExplanation,
      ``,

      `MODEL CONTEXT`,
      `The Merchant Risk Shield Chargeback Win Predictor is a Random Forest`,
      `model trained on ${disputeCount} historical disputes. Its probability`,
      `score is a predictive signal and should be considered alongside the`,
      `available case evidence.`,
      ``,

      `Potential transaction exposure: ${amount}`,
      ``,

      `Regards,`,
      `Merchant Risk Shield — Automated Evidence Desk`,
    ];

    return lines.join("\n");
  }


  // -------------------------------------------------------------------------
  // 5. MEDIUM PROBABILITY
  // -------------------------------------------------------------------------

  if (probability < 70) {
    const lines = [
      `Subject: Chargeback Review — Dispute ${d.dispute_id} / Order ${d.order_id}`,
      ``,
      `To the Merchant Dispute Resolution Team,`,
      ``,

      `We have reviewed chargeback ${d.dispute_id} against order ${d.order_id}`,
      `(value ${amount}) filed under the reason "${d.reason}."`,
      ``,

      `RISK ASSESSMENT`,
      `${decisionTitle}: ${probability}% predicted win probability.`,
      `Recommended action: ${decision}`,
      ``,

      `AVAILABLE CASE EVIDENCE`,
      ...evidenceLines,
      ``,

      `RECOMMENDATION`,
      decisionExplanation,
      ``,

      `The Chargeback Win Predictor (Random Forest, trained on`,
      `${disputeCount} historical disputes) indicates an uncertain outcome.`,
      `Additional documentation should be reviewed or obtained before`,
      `a final representment decision is made.`,
      ``,

      `Transaction value under review: ${amount}`,
      ``,

      `Regards,`,
      `Merchant Risk Shield — Automated Evidence Desk`,
    ];

    return lines.join("\n");
  }


  // -------------------------------------------------------------------------
  // 6. HIGH PROBABILITY
  // -------------------------------------------------------------------------

  const lines = [
    `Subject: Chargeback Representment — Dispute ${d.dispute_id} / Order ${d.order_id}`,
    ``,

    `To the Card Issuing Bank / Dispute Resolution Team,`,
    ``,

    `We are writing to formally contest the chargeback filed against order ${d.order_id}`,
    `(value ${amount}) on the grounds of "${d.reason}."`,
    ``,

    `REPRESENTMENT BASIS`,
    ``,

    ...evidenceLines,
    ``,

    `RISK ASSESSMENT`,
    `The Merchant Risk Shield Chargeback Win Predictor (Random Forest, trained`,
    `on ${disputeCount} historical disputes) assigns this case a ${probability}%`,
    `predicted win probability based on the available case evidence.`,
    ``,

    `RECOMMENDED RESOLUTION`,
    `The available evidence strongly supports the merchant's position.`,
    `We respectfully request reversal of the chargeback and reinstatement`,
    `of ${amount}.`,
    ``,

    `The supporting documentation referenced above can be provided as part`,
    `of the representment package for dispute review.`,
    ``,

    `Regards,`,
    `Merchant Risk Shield — Automated Evidence Desk`,
  ];

  return lines.join("\n");
}


// ---------------------------------------------------------------------------
// Routes
// ---------------------------------------------------------------------------

app.get("/api/metrics", (req, res) => {
  res.json({
    model1: state.model1,
    model2: state.model2,
    datasetSizes: state.datasetSizes,
    savedTrend: state.savedTrend,
  });
});


app.get("/api/orders", (req, res) => {
  res.json(state.orders);
});


app.post("/api/orders/:id/verify", (req, res) => {
  const o = state.orders.find(
    (x) => x.order_id === req.params.id
  );

  if (!o) {
    return res.status(404).json({
      error: "order not found",
    });
  }

  o.status = "verified";

  res.json(o);
});


app.post("/api/orders/:id/block", (req, res) => {
  const o = state.orders.find(
    (x) => x.order_id === req.params.id
  );

  if (!o) {
    return res.status(404).json({
      error: "order not found",
    });
  }

  o.status = "blocked_cod";

  res.json(o);
});


app.get("/api/disputes", (req, res) => {
  res.json(state.disputes);
});


app.post("/api/disputes/:id/generate-letter", (req, res) => {
  const d = state.disputes.find(
    (x) => x.dispute_id === req.params.id
  );

  if (!d) {
    return res.status(404).json({
      error: "dispute not found",
    });
  }

  d.letter = buildLetter(d);

  d.status = "letter_ready";

  res.json(d);
});


app.post("/api/disputes/:id/submit", (req, res) => {
  const d = state.disputes.find(
    (x) => x.dispute_id === req.params.id
  );

  if (!d) {
    return res.status(404).json({
      error: "dispute not found",
    });
  }

  if (!d.letter) {
    return res.status(400).json({
      error: "generate the letter before submitting",
    });
  }

  d.status = "submitted";

  res.json(d);
});


app.get("/api/health", (req, res) => {
  res.json({
    ok: true,
    orders: state.orders.length,
    disputes: state.disputes.length,
  });
});


const PORT = process.env.PORT || 4000;

app.listen(PORT, () =>
  console.log(
    `Merchant Risk Shield API running on http://localhost:${PORT}`
  )
);
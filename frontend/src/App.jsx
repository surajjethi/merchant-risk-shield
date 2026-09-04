import React, { useState, useMemo, useEffect } from "react";
import {
  Shield, Search, CheckCircle2, Ban, FileText, Send, X, Sparkles,
  ArrowUpRight, Truck, Wifi, MessageSquare, Clock, Cpu, Database,
  Layers, Scale, Zap, Plus, Loader2, WifiOff, Lock, Eye, EyeOff, LogOut,
} from "lucide-react";
import {
  ResponsiveContainer, AreaChart, Area, XAxis, Tooltip,
} from "recharts";

/* ============================================================================
   DATA — snapshot of ml-pipeline/export.json, produced by:
     generate_data.py  -> 6,000 synthetic orders / 1,400 synthetic disputes
     train_models.py   -> XGBoost (return risk) + Random Forest (win predictor)
   precision/recall/F1/AUC/confusion matrices/feature importances below are
   the real sklearn/xgboost evaluation output, not authored numbers.
============================================================================ */

const DATASET_SIZES = { orders: 6000, disputes: 1400 };

// FastAPI scoring microservice (ml-pipeline/scoring_service.py) — run it with
// `uvicorn scoring_service:app --reload --port 8000` to make the "Score live"
// panels below hit the real trained XGBoost / Random Forest models.
const SCORING_URL = "https://merchant-risk-shield.onrender.com";

const MODEL1 = {
  name: "Return Risk Scorer",
  algo: "XGBoost",
  detail: "250 trees · depth 4 · lr 0.08",
  metrics: { precision: 68.1, recall: 58.4, f1: 62.9, auc: 75.0 },
  confusion: [[630, 187], [284, 399]],
  importances: [
    { f: "hist_return_rate", v: 0.1382 },
    { f: "device_ip_mismatch", v: 0.1373 },
    { f: "prior_disputes", v: 0.1068 },
    { f: "is_cod", v: 0.0988 },
    { f: "delivery_pincode_risk", v: 0.0454 },
  ],
};

const MODEL2 = {
  name: "Chargeback Win Predictor",
  algo: "Random Forest",
  detail: "300 trees · depth 7 · balanced",
  metrics: { precision: 86.4, recall: 81.5, f1: 83.9, auc: 85.4 },
  confusion: [[87, 30], [43, 190]],
  importances: [
    { f: "has_signed_pod", v: 0.3156 },
    { f: "has_delivery_proof", v: 0.2025 },
    { f: "tracking_valid", v: 0.13 },
    { f: "order_value", v: 0.0861 },
    { f: "days_to_file", v: 0.0724 },
  ],
};

const SAVED_TREND = [
  { month: "Mar", k: 9.1 }, { month: "Apr", k: 8.38 }, { month: "May", k: 11.18 },
  { month: "Jun", k: 11.96 }, { month: "Jul", k: 13.35 }, { month: "Aug", k: 14.84 },
];

const RAW_ORDERS = [
  { order_id: "ORD-11925", customer: "Janaki Zacharia", category: "Electronics", order_value: 4680.31, is_cod: 1, past_orders: 18, hist_return_rate: 0.531, risk_score: 91 },
  { order_id: "ORD-13346", customer: "Jagvi Jhaveri", category: "Electronics", order_value: 8070.23, is_cod: 0, past_orders: 31, hist_return_rate: 0.53, risk_score: 87 },
  { order_id: "ORD-14493", customer: "Harsh Virk", category: "Mobiles", order_value: 5220.01, is_cod: 1, past_orders: 2, hist_return_rate: 0.383, risk_score: 80 },
  { order_id: "ORD-14316", customer: "Omkaar Nagy", category: "Home & Kitchen", order_value: 9531.48, is_cod: 1, past_orders: 10, hist_return_rate: 0.417, risk_score: 78 },
  { order_id: "ORD-10616", customer: "Anika Magar", category: "Electronics", order_value: 6588.54, is_cod: 1, past_orders: 4, hist_return_rate: 0.364, risk_score: 76 },
  { order_id: "ORD-15073", customer: "Guneet Sarna", category: "Mobiles", order_value: 6401.37, is_cod: 0, past_orders: 13, hist_return_rate: 0.481, risk_score: 72 },
  { order_id: "ORD-11625", customer: "Lajita Trivedi", category: "Mobiles", order_value: 10767.15, is_cod: 1, past_orders: 6, hist_return_rate: 0.481, risk_score: 70 },
  { order_id: "ORD-14449", customer: "Agastya Morar", category: "Home & Kitchen", order_value: 3752.87, is_cod: 1, past_orders: 10, hist_return_rate: 0.361, risk_score: 65 },
  { order_id: "ORD-15559", customer: "Owen Naidu", category: "Home & Kitchen", order_value: 7464.53, is_cod: 1, past_orders: 7, hist_return_rate: 0.297, risk_score: 65 },
  { order_id: "ORD-12352", customer: "Chavvi Keer", category: "Electronics", order_value: 13416.65, is_cod: 1, past_orders: 26, hist_return_rate: 0.342, risk_score: 64 },
  { order_id: "ORD-10179", customer: "Upma Bahl", category: "Home & Kitchen", order_value: 3240.15, is_cod: 0, past_orders: 17, hist_return_rate: 0.321, risk_score: 62 },
  { order_id: "ORD-13213", customer: "Pranit Chander", category: "Fitness", order_value: 5701.09, is_cod: 0, past_orders: 17, hist_return_rate: 0.191, risk_score: 59 },
  { order_id: "ORD-15461", customer: "Liam Bera", category: "Mobiles", order_value: 5204.58, is_cod: 0, past_orders: 7, hist_return_rate: 0.408, risk_score: 56 },
  { order_id: "ORD-13806", customer: "Gautam Konda", category: "Mobiles", order_value: 6061.22, is_cod: 1, past_orders: 10, hist_return_rate: 0.368, risk_score: 54 },
  { order_id: "ORD-15624", customer: "Eesha Balay", category: "Fashion", order_value: 8038.41, is_cod: 1, past_orders: 4, hist_return_rate: 0.346, risk_score: 53 },
  { order_id: "ORD-13279", customer: "Ladli Madan", category: "Beauty", order_value: 9087.77, is_cod: 0, past_orders: 21, hist_return_rate: 0.211, risk_score: 48 },
  { order_id: "ORD-15215", customer: "Girik Raja", category: "Fashion", order_value: 7491.46, is_cod: 0, past_orders: 7, hist_return_rate: 0.316, risk_score: 47 },
  { order_id: "ORD-12604", customer: "Chasmum Maharaj", category: "Electronics", order_value: 6186.57, is_cod: 0, past_orders: 13, hist_return_rate: 0.437, risk_score: 43 },
  { order_id: "ORD-11208", customer: "Widisha Badami", category: "Home & Kitchen", order_value: 2646.15, is_cod: 1, past_orders: 17, hist_return_rate: 0.253, risk_score: 40 },
  { order_id: "ORD-10905", customer: "Adweta Kale", category: "Electronics", order_value: 2766.27, is_cod: 1, past_orders: 12, hist_return_rate: 0.283, risk_score: 37 },
  { order_id: "ORD-13002", customer: "Jyoti Nair", category: "Mobiles", order_value: 6451.14, is_cod: 1, past_orders: 8, hist_return_rate: 0.196, risk_score: 23 },
  { order_id: "ORD-15883", customer: "Jasmit Bala", category: "Electronics", order_value: 4779.44, is_cod: 0, past_orders: 13, hist_return_rate: 0.192, risk_score: 22 },
  { order_id: "ORD-14320", customer: "Jagdish Kapur", category: "Mobiles", order_value: 2969.24, is_cod: 0, past_orders: 7, hist_return_rate: 0.176, risk_score: 14 },
  { order_id: "ORD-10729", customer: "Anay Thakkar", category: "Electronics", order_value: 10851.82, is_cod: 0, past_orders: 8, hist_return_rate: 0.22, risk_score: 8 },
].map((o) => ({ ...o, status: "pending" }));

const RAW_DISPUTES = [
  { dispute_id: "DSP-2300", order_id: "ORD-12929", reason: "Item not received", order_value: 2742.31, has_signed_pod: 1, ip_match: 1, tracking_valid: 1, comm_logs: 2, win_probability: 91 },
  { dispute_id: "DSP-3137", order_id: "ORD-15251", reason: "Duplicate charge", order_value: 7887.4, has_signed_pod: 1, ip_match: 1, tracking_valid: 1, comm_logs: 1, win_probability: 88 },
  { dispute_id: "DSP-3250", order_id: "ORD-12807", reason: "Item not received", order_value: 6766.55, has_signed_pod: 1, ip_match: 1, tracking_valid: 1, comm_logs: 0, win_probability: 85 },
  { dispute_id: "DSP-2789", order_id: "ORD-11227", reason: "Item not as described", order_value: 10087.61, has_signed_pod: 1, ip_match: 1, tracking_valid: 1, comm_logs: 0, win_probability: 83 },
  { dispute_id: "DSP-2098", order_id: "ORD-10440", reason: "Item not received", order_value: 1207.01, has_signed_pod: 1, ip_match: 1, tracking_valid: 0, comm_logs: 2, win_probability: 82 },
  { dispute_id: "DSP-2828", order_id: "ORD-11953", reason: "Item not received", order_value: 2967.94, has_signed_pod: 1, ip_match: 0, tracking_valid: 1, comm_logs: 2, win_probability: 77 },
  { dispute_id: "DSP-2630", order_id: "ORD-12873", reason: "Item not received", order_value: 2884.11, has_signed_pod: 0, ip_match: 1, tracking_valid: 1, comm_logs: 0, win_probability: 68 },
  { dispute_id: "DSP-2721", order_id: "ORD-14728", reason: "Item not as described", order_value: 3823.15, has_signed_pod: 1, ip_match: 0, tracking_valid: 1, comm_logs: 1, win_probability: 65 },
  { dispute_id: "DSP-2017", order_id: "ORD-12380", reason: "Duplicate charge", order_value: 18509.86, has_signed_pod: 1, ip_match: 0, tracking_valid: 0, comm_logs: 0, win_probability: 60 },
  { dispute_id: "DSP-3356", order_id: "ORD-15720", reason: "Duplicate charge", order_value: 3052.05, has_signed_pod: 0, ip_match: 1, tracking_valid: 0, comm_logs: 1, win_probability: 31 },
  { dispute_id: "DSP-3002", order_id: "ORD-11082", reason: "Item not received", order_value: 1656.02, has_signed_pod: 0, ip_match: 1, tracking_valid: 0, comm_logs: 0, win_probability: 25 },
  { dispute_id: "DSP-2101", order_id: "ORD-11600", reason: "Item not as described", order_value: 8923.0, has_signed_pod: 0, ip_match: 1, tracking_valid: 0, comm_logs: 3, win_probability: 24 },
  { dispute_id: "DSP-2810", order_id: "ORD-12867", reason: "Defective item", order_value: 4495.09, has_signed_pod: 0, ip_match: 0, tracking_valid: 0, comm_logs: 0, win_probability: 2 },
].map((d) => ({ ...d, status: "pending", letter: null }));

/* ============================================================================
   Theme — real color/type tokens (no Tailwind arbitrary values anywhere;
   this environment only ships pre-defined utility classes, so every custom
   color/size below is applied via inline style or the injected stylesheet).
============================================================================ */

const C = {
  bg: "#0A0D12",
  panel: "#12151C",
  panelAlt: "#161A22",
  sunken: "#0E1015",
  border: "#232833",
  borderHi: "#333A48",
  hair: "#1E222C",
  textHi: "#EDEFF3",
  textMid: "#C7CCD6",
  textLo: "#8A93A3",
  textFaint: "#5B6474",
  amber: "#E7A93C",
  amberBg: "#2A2312",
  teal: "#34D8B0",
  tealBg: "#132622",
  coral: "#FF6259",
  coralBg: "#251413",
  violet: "#9C8CFF",
  violetBg: "#201D33",
};

const mono = { fontFamily: "'IBM Plex Mono', ui-monospace, monospace" };
const display = { fontFamily: "'Space Grotesk', ui-sans-serif, system-ui" };

/* ============================================================================
   Helpers
============================================================================ */

const fmtINR = (n) => `\u20B9${Number(n).toLocaleString("en-IN", { maximumFractionDigits: 0 })}`;

function riskTone(score) {
  if (score >= 65) return { hex: C.coral, label: "High" };
  if (score >= 35) return { hex: C.amber, label: "Medium" };
  return { hex: C.teal, label: "Low" };
}

function buildLetter(d) {
  const lines = [
    `RE: Chargeback ${d.dispute_id} — Order ${d.order_id}`,
    ``,
    `To the Card Issuing Bank, Dispute Resolution Desk,`,
    ``,
    `We contest the chargeback filed against order ${d.order_id} (${fmtINR(d.order_value)}),`,
    `raised on the grounds of "${d.reason}."`,
    ``,
    `Evidence on file:`,
    d.has_signed_pod ? `  - Signed proof of delivery on record for this shipment.` : `  - No signed proof-of-delivery could be located.`,
    d.tracking_valid ? `  - Courier tracking history verified end to end.` : `  - Tracking history is incomplete.`,
    d.ip_match ? `  - Order device/IP matches the customer's known account activity.` : `  - Device/IP match was inconclusive.`,
    `  - ${d.comm_logs} prior support communication(s), with no non-delivery report before filing.`,
    ``,
    `Chargeback Win Predictor (Random Forest, ${DATASET_SIZES.disputes.toLocaleString("en-IN")} historical disputes)`,
    `scores this case at ${d.win_probability}% win probability.`,
    ``,
    `We request reversal of this chargeback and reinstatement of ${fmtINR(d.order_value)}.`,
    ``,
    `Merchant Risk Shield — Automated Evidence Desk`,
  ];
  return lines.join("\n");
}

/* ============================================================================
   Small building blocks
============================================================================ */

function RadialGauge({ value, size = 76, stroke = 7, color, sub }) {
  const r = (size - stroke) / 2;
  const c = 2 * Math.PI * r;
  const [animated, setAnimated] = useState(0);
  useEffect(() => {
    const t = setTimeout(() => setAnimated(value), 120);
    return () => clearTimeout(t);
  }, [value]);
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
      <svg width={size} height={size} style={{ flexShrink: 0, transform: "rotate(-90deg)" }}>
        <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke={C.hair} strokeWidth={stroke} />
        <circle
          cx={size / 2} cy={size / 2} r={r} fill="none" stroke={color} strokeWidth={stroke}
          strokeDasharray={c} strokeDashoffset={c - (animated / 100) * c} strokeLinecap="round"
          style={{ transition: "stroke-dashoffset 1s cubic-bezier(.4,0,.2,1)" }}
        />
      </svg>
      <div>
        <div style={{ ...mono, fontSize: 19, lineHeight: 1, fontWeight: 600, color: C.textHi }}>{value}</div>
        {sub && <div style={{ fontSize: 10, color: C.textFaint, marginTop: 4 }}>{sub}</div>}
      </div>
    </div>
  );
}

function Pulse() {
  return (
    <span className="mrs-pulse-wrap">
      <span className="mrs-pulse-ping" />
      <span className="mrs-pulse-dot" />
    </span>
  );
}

function SectionLabel({ children, icon }) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 11, letterSpacing: "0.06em", color: "#7C8598", fontWeight: 500 }}>
      {icon}{children}
    </div>
  );
}

/* ============================================================================
   Top overview strip
============================================================================ */

function Overview({ orders, disputes }) {
  const highRisk = orders.filter((o) => o.risk_score >= 65).length;
  const winnable = disputes.filter((d) => d.win_probability >= 65).length;
  const latest = SAVED_TREND[SAVED_TREND.length - 1].k;
  const first = SAVED_TREND[0].k;
  const growth = Math.round(((latest - first) / first) * 100);

  return (
    <div style={{ display: "grid", gridTemplateColumns: "1.7fr 1fr 1fr", gap: 16, marginBottom: 24 }}>
      <div style={{
        borderRadius: 12, border: `1px solid ${C.border}`, padding: 20,
        background: `linear-gradient(135deg, #151922 0%, #11141B 100%)`,
      }}>
        <SectionLabel icon={<Sparkles size={12} color={C.amber} />}>Estimated recovery, trailing 6 months</SectionLabel>
        <div style={{ display: "flex", alignItems: "baseline", gap: 8, marginTop: 8 }}>
          <span style={{ ...mono, fontSize: 34, fontWeight: 600, color: C.textHi }}>₹{latest}K</span>
          <span style={{ display: "flex", alignItems: "center", gap: 2, fontSize: 12, fontWeight: 500, color: C.teal }}>
            <ArrowUpRight size={13} /> {growth}%
          </span>
        </div>
        <div style={{ fontSize: 11.5, color: C.textFaint, marginTop: 2 }}>
          across {DATASET_SIZES.orders.toLocaleString("en-IN")} orders + {DATASET_SIZES.disputes.toLocaleString("en-IN")} disputes scored
        </div>
        <div style={{ height: 74, marginTop: 12, marginLeft: -8, marginRight: -8 }}>
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={SAVED_TREND} margin={{ top: 4, left: 8, right: 8, bottom: 0 }}>
              <defs>
                <linearGradient id="savedFill" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor={C.amber} stopOpacity={0.35} />
                  <stop offset="100%" stopColor={C.amber} stopOpacity={0} />
                </linearGradient>
              </defs>
              <XAxis dataKey="month" hide />
              <Tooltip
                cursor={false}
                contentStyle={{ background: C.panelAlt, border: `1px solid ${C.border}`, borderRadius: 6, fontSize: 11 }}
                labelStyle={{ color: C.textLo }}
                formatter={(v) => [`₹${v}K`, "Saved"]}
              />
              <Area type="monotone" dataKey="k" stroke={C.amber} strokeWidth={2} fill="url(#savedFill)" />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div style={{ borderRadius: 12, border: `1px solid ${C.border}`, background: C.panel, padding: 16, display: "flex", flexDirection: "column", justifyContent: "space-between" }}>
        <SectionLabel icon={<Ban size={12} color={C.coral} />}>High-risk orders</SectionLabel>
        <RadialGauge value={highRisk} size={64} stroke={6} color={C.coral} sub={`of ${orders.length} shown`} />
      </div>
      <div style={{ borderRadius: 12, border: `1px solid ${C.border}`, background: C.panel, padding: 16, display: "flex", flexDirection: "column", justifyContent: "space-between" }}>
        <SectionLabel icon={<Scale size={12} color={C.violet} />}>Winnable disputes</SectionLabel>
        <RadialGauge value={winnable} size={64} stroke={6} color={C.violet} sub={`of ${disputes.length} shown`} />
      </div>
    </div>
  );
}

/* ============================================================================
   Live scoring panel — calls the real FastAPI microservice
   (ml-pipeline/scoring_service.py) which loads the actual trained
   XGBoost / Random Forest .pkl files and scores whatever you type in.
============================================================================ */

const ORDER_CATEGORIES = ["Electronics", "Fashion", "Fitness", "Home & Kitchen", "Beauty", "Mobiles"];
const DISPUTE_REASONS = ["Item not received", "Item not as described", "Transaction not recognised", "Duplicate charge", "Defective item"];

function fieldStyle() {
  return {
    width: "100%", background: C.sunken, border: `1px solid ${C.border}`, borderRadius: 6,
    padding: "6px 8px", fontSize: 12, color: C.textHi, outline: "none", boxSizing: "border-box",
  };
}
function fieldLabel(children) {
  return <div style={{ fontSize: 10, color: C.textFaint, marginBottom: 4 }}>{children}</div>;
}

function LiveOrderScorer({ onAdd }) {
  const [open, setOpen] = useState(false);
  const [form, setForm] = useState({
    order_value: 8500, category: "Electronics", is_cod: true, past_orders: 12,
    hist_return_rate: 0.4, device_ip_mismatch: false, delivery_pincode_risk: 0.3,
  });
  const [state, setState] = useState({ loading: false, result: null, error: null });

  async function score() {
    setState({ loading: true, result: null, error: null });
    try {
      const res = await fetch(`${SCORING_URL}/score-order`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          ...form,
          days_since_signup: 180, prior_disputes: 0, night_order: false, coupon_used: false,
        }),
      });
      if (!res.ok) throw new Error("service returned an error");
      const data = await res.json();
      setState({ loading: false, result: data, error: null });
    } catch (e) {
      setState({ loading: false, result: null, error: "Scoring service unreachable — run `uvicorn scoring_service:app --port 8000` in ml-pipeline/" });
    }
  }

  function addToTable() {
    if (!state.result) return;
    onAdd({
      order_id: `ORD-LIVE-${Math.floor(Math.random() * 9000 + 1000)}`,
      customer: "New customer (live scored)",
      category: form.category,
      order_value: Number(form.order_value),
      is_cod: form.is_cod ? 1 : 0,
      past_orders: Number(form.past_orders),
      hist_return_rate: Number(form.hist_return_rate),
      risk_score: state.result.risk_score,
      status: "pending",
      live: true,
    });
    setState({ loading: false, result: null, error: null });
    setOpen(false);
  }

  const tone = state.result ? riskTone(state.result.risk_score) : null;

  return (
    <div style={{ borderRadius: 12, border: `1px solid ${C.border}`, background: C.panel, marginBottom: 16, overflow: "hidden" }}>
      <button onClick={() => setOpen((o) => !o)} style={{
        width: "100%", display: "flex", alignItems: "center", justifyContent: "space-between",
        padding: "12px 16px", background: "transparent", border: "none", cursor: "pointer",
      }}>
        <span style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 12.5, fontWeight: 600, color: C.textHi }}>
          <Zap size={14} color={C.amber} /> Score a new order live (real XGBoost model)
        </span>
        <Plus size={16} color={C.textFaint} style={{ transform: open ? "rotate(45deg)" : "none", transition: "transform .15s" }} />
      </button>
      {open && (
        <div style={{ padding: "0 16px 16px", borderTop: `1px solid ${C.hair}`, paddingTop: 16 }}>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 10 }}>
            <div>
              {fieldLabel("Order value (₹)")}
              <input type="number" style={fieldStyle()} value={form.order_value}
                onChange={(e) => setForm({ ...form, order_value: e.target.value })} />
            </div>
            <div>
              {fieldLabel("Category")}
              <select style={fieldStyle()} value={form.category}
                onChange={(e) => setForm({ ...form, category: e.target.value })}>
                {ORDER_CATEGORIES.map((c) => <option key={c} value={c}>{c}</option>)}
              </select>
            </div>
            <div>
              {fieldLabel("Payment")}
              <select style={fieldStyle()} value={form.is_cod ? "cod" : "prepaid"}
                onChange={(e) => setForm({ ...form, is_cod: e.target.value === "cod" })}>
                <option value="cod">COD</option>
                <option value="prepaid">Prepaid</option>
              </select>
            </div>
            <div>
              {fieldLabel("Past orders")}
              <input type="number" style={fieldStyle()} value={form.past_orders}
                onChange={(e) => setForm({ ...form, past_orders: e.target.value })} />
            </div>
            <div>
              {fieldLabel(`Historical return rate — ${Math.round(form.hist_return_rate * 100)}%`)}
              <input type="range" min="0" max="1" step="0.01" style={{ width: "100%" }} value={form.hist_return_rate}
                onChange={(e) => setForm({ ...form, hist_return_rate: e.target.value })} />
            </div>
            <div>
              {fieldLabel(`Delivery pincode risk — ${Math.round(form.delivery_pincode_risk * 100)}%`)}
              <input type="range" min="0" max="1" step="0.01" style={{ width: "100%" }} value={form.delivery_pincode_risk}
                onChange={(e) => setForm({ ...form, delivery_pincode_risk: e.target.value })} />
            </div>
          </div>

          <label style={{ display: "flex", alignItems: "center", gap: 6, marginTop: 12, fontSize: 11.5, color: C.textLo, cursor: "pointer" }}>
            <input type="checkbox" checked={form.device_ip_mismatch}
              onChange={(e) => setForm({ ...form, device_ip_mismatch: e.target.checked })} />
            Device / IP mismatch flagged
          </label>

          <div style={{ display: "flex", alignItems: "center", gap: 12, marginTop: 14 }}>
            <button onClick={score} disabled={state.loading} className="mrs-primary-btn" style={{
              display: "flex", alignItems: "center", gap: 6, padding: "8px 16px", borderRadius: 8,
              background: C.amber, color: "#12151C", fontSize: 12, fontWeight: 600, border: "none",
              cursor: state.loading ? "default" : "pointer", opacity: state.loading ? 0.7 : 1,
            }}>
              {state.loading ? <Loader2 size={13} className="mrs-spin" /> : <Zap size={13} />}
              {state.loading ? "Scoring…" : "Score live"}
            </button>

            {state.result && (
              <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                <span style={{ ...mono, fontSize: 22, fontWeight: 700, color: tone.hex }}>{state.result.risk_score}</span>
                <span style={{ fontSize: 11, color: tone.hex, fontWeight: 500 }}>{tone.label} risk</span>
                <button onClick={addToTable} style={{
                  fontSize: 11, color: C.textLo, background: "transparent", border: `1px solid ${C.border}`,
                  borderRadius: 6, padding: "5px 10px", cursor: "pointer",
                }}>+ Add to Orders table</button>
              </div>
            )}
          </div>

          {state.error && (
            <div style={{ display: "flex", alignItems: "center", gap: 6, marginTop: 10, fontSize: 11, color: C.coral }}>
              <WifiOff size={12} /> {state.error}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function LiveDisputeScorer({ onAdd }) {
  const [open, setOpen] = useState(false);
  const [form, setForm] = useState({
    order_value: 6500, reason: "Item not received", has_signed_pod: true,
    ip_match: true, tracking_valid: true, comm_logs: 1,
  });
  const [state, setState] = useState({ loading: false, result: null, error: null });

  async function score() {
    setState({ loading: true, result: null, error: null });
    try {
      const res = await fetch(`${SCORING_URL}/score-dispute`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          ...form, has_delivery_proof: form.has_signed_pod, days_to_file: 5, prior_customer_disputes: 0,
        }),
      });
      if (!res.ok) throw new Error("service returned an error");
      const data = await res.json();
      setState({ loading: false, result: data, error: null });
    } catch (e) {
      setState({ loading: false, result: null, error: "Scoring service unreachable — run `uvicorn scoring_service:app --port 8000` in ml-pipeline/" });
    }
  }

  function addToTable() {
    if (!state.result) return;
    onAdd({
      dispute_id: `DSP-LIVE-${Math.floor(Math.random() * 9000 + 1000)}`,
      order_id: `ORD-LIVE-${Math.floor(Math.random() * 9000 + 1000)}`,
      reason: form.reason,
      order_value: Number(form.order_value),
      has_signed_pod: form.has_signed_pod ? 1 : 0,
      ip_match: form.ip_match ? 1 : 0,
      tracking_valid: form.tracking_valid ? 1 : 0,
      comm_logs: Number(form.comm_logs),
      win_probability: state.result.win_probability,
      status: "pending", letter: null, live: true,
    });
    setState({ loading: false, result: null, error: null });
    setOpen(false);
  }

  const winColor = state.result ? (state.result.win_probability >= 65 ? C.teal : state.result.win_probability >= 35 ? C.amber : C.coral) : null;

  return (
    <div style={{ borderRadius: 12, border: `1px solid ${C.border}`, background: C.panel, marginBottom: 16, overflow: "hidden" }}>
      <button onClick={() => setOpen((o) => !o)} style={{
        width: "100%", display: "flex", alignItems: "center", justifyContent: "space-between",
        padding: "12px 16px", background: "transparent", border: "none", cursor: "pointer",
      }}>
        <span style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 12.5, fontWeight: 600, color: C.textHi }}>
          <Zap size={14} color={C.violet} /> Score a new dispute live (real Random Forest model)
        </span>
        <Plus size={16} color={C.textFaint} style={{ transform: open ? "rotate(45deg)" : "none", transition: "transform .15s" }} />
      </button>
      {open && (
        <div style={{ padding: "0 16px 16px", borderTop: `1px solid ${C.hair}`, paddingTop: 16 }}>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 10 }}>
            <div>
              {fieldLabel("Order value (₹)")}
              <input type="number" style={fieldStyle()} value={form.order_value}
                onChange={(e) => setForm({ ...form, order_value: e.target.value })} />
            </div>
            <div style={{ gridColumn: "span 2" }}>
              {fieldLabel("Reason")}
              <select style={fieldStyle()} value={form.reason}
                onChange={(e) => setForm({ ...form, reason: e.target.value })}>
                {DISPUTE_REASONS.map((r) => <option key={r} value={r}>{r}</option>)}
              </select>
            </div>
            <div>
              {fieldLabel("Comm logs on file")}
              <input type="number" style={fieldStyle()} value={form.comm_logs}
                onChange={(e) => setForm({ ...form, comm_logs: e.target.value })} />
            </div>
          </div>

          <div style={{ display: "flex", gap: 16, marginTop: 12, flexWrap: "wrap" }}>
            {[
              ["has_signed_pod", "Signed proof of delivery"],
              ["ip_match", "IP / device match"],
              ["tracking_valid", "Tracking verified"],
            ].map(([k, label]) => (
              <label key={k} style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 11.5, color: C.textLo, cursor: "pointer" }}>
                <input type="checkbox" checked={form[k]} onChange={(e) => setForm({ ...form, [k]: e.target.checked })} />
                {label}
              </label>
            ))}
          </div>

          <div style={{ display: "flex", alignItems: "center", gap: 12, marginTop: 14 }}>
            <button onClick={score} disabled={state.loading} className="mrs-primary-btn" style={{
              display: "flex", alignItems: "center", gap: 6, padding: "8px 16px", borderRadius: 8,
              background: C.violet, color: "#12151C", fontSize: 12, fontWeight: 600, border: "none",
              cursor: state.loading ? "default" : "pointer", opacity: state.loading ? 0.7 : 1,
            }}>
              {state.loading ? <Loader2 size={13} className="mrs-spin" /> : <Zap size={13} />}
              {state.loading ? "Scoring…" : "Score live"}
            </button>

            {state.result && (
              <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                <span style={{ ...mono, fontSize: 22, fontWeight: 700, color: winColor }}>{state.result.win_probability}%</span>
                <span style={{ fontSize: 11, color: winColor, fontWeight: 500 }}>win probability</span>
                <button onClick={addToTable} style={{
                  fontSize: 11, color: C.textLo, background: "transparent", border: `1px solid ${C.border}`,
                  borderRadius: 6, padding: "5px 10px", cursor: "pointer",
                }}>+ Add to Disputes board</button>
              </div>
            )}
          </div>

          {state.error && (
            <div style={{ display: "flex", alignItems: "center", gap: 6, marginTop: 10, fontSize: 11, color: C.coral }}>
              <WifiOff size={12} /> {state.error}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

/* ============================================================================
   Orders tab
============================================================================ */

function OrdersTab({ orders, setOrders }) {
  const [q, setQ] = useState("");
  const [band, setBand] = useState("all");

  const filtered = useMemo(() => orders.filter((o) => {
    const t = riskTone(o.risk_score).label.toLowerCase();
    if (band !== "all" && t !== band) return false;
    if (q && !(`${o.customer} ${o.order_id} ${o.category}`.toLowerCase().includes(q.toLowerCase()))) return false;
    return true;
  }), [orders, band, q]);

  function act(id, status) {
    setOrders((prev) => prev.map((o) => (o.order_id === id ? { ...o, status } : o)));
  }

  return (
    <>
    <LiveOrderScorer onAdd={(o) => setOrders((prev) => [o, ...prev])} />
    <div style={{ borderRadius: 12, border: `1px solid ${C.border}`, background: C.panel, overflow: "hidden" }}>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", padding: "12px 16px", borderBottom: `1px solid ${C.hair}`, gap: 12, flexWrap: "wrap" }}>
        <div style={{ display: "flex", gap: 4 }}>
          {["all", "high", "medium", "low"].map((b) => (
            <button key={b} onClick={() => setBand(b)} className="mrs-filter-btn" style={{
              padding: "5px 10px", borderRadius: 6, fontSize: 11.5, fontWeight: 500, cursor: "pointer",
              background: band === b ? "#1E232E" : "transparent",
              border: `1px solid ${band === b ? C.borderHi : "transparent"}`,
              color: band === b ? C.textHi : "#7C8598",
            }}>
              {b === "all" ? "All" : b[0].toUpperCase() + b.slice(1)}
            </button>
          ))}
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 8, background: C.sunken, border: `1px solid ${C.border}`, borderRadius: 6, padding: "6px 10px", width: 224 }}>
          <Search size={13} color={C.textFaint} />
          <input value={q} onChange={(e) => setQ(e.target.value)} placeholder="Search orders"
            style={{ background: "transparent", outline: "none", border: "none", fontSize: 12, width: "100%", color: C.textHi }} />
        </div>
      </div>

      <div style={{ overflow: "auto", maxHeight: 520 }}>
        <table style={{ width: "100%", fontSize: 12.5, borderCollapse: "collapse" }}>
          <thead style={{ position: "sticky", top: 0, background: C.panel, zIndex: 10 }}>
            <tr style={{ textAlign: "left", fontSize: 10.5, letterSpacing: "0.02em", color: C.textFaint, borderBottom: `1px solid ${C.hair}` }}>
              <th style={{ padding: "8px 16px", fontWeight: 500 }}>Order</th>
              <th style={{ padding: "8px 12px", fontWeight: 500 }}>Customer</th>
              <th style={{ padding: "8px 12px", fontWeight: 500 }}>Category</th>
              <th style={{ padding: "8px 12px", fontWeight: 500, textAlign: "right" }}>Value</th>
              <th style={{ padding: "8px 12px", fontWeight: 500 }}>Pay</th>
              <th style={{ padding: "8px 12px", fontWeight: 500, width: 128 }}>Risk</th>
              <th style={{ padding: "8px 12px", fontWeight: 500 }}>Status</th>
              <th style={{ padding: "8px 16px", fontWeight: 500, textAlign: "right" }}>Action</th>
            </tr>
          </thead>
          <tbody>
            {filtered.map((o) => {
              const tone = riskTone(o.risk_score);
              return (
                <tr key={o.order_id} className="mrs-row" style={{ borderBottom: `1px solid #161920` }}>
                  <td style={{ padding: "10px 16px", ...mono, fontSize: 11.5, color: C.textMid }}>
                    {o.order_id}
                    {o.live && <span style={{ marginLeft: 6, fontSize: 8.5, fontWeight: 700, color: C.amber, background: C.amberBg, padding: "1px 5px", borderRadius: 4 }}>LIVE</span>}
                  </td>
                  <td style={{ padding: "10px 12px", color: C.textHi }}>
                    {o.customer}
                    <div style={{ ...mono, fontSize: 10.5, color: C.textFaint }}>{o.past_orders} ord · {(o.hist_return_rate * 100).toFixed(0)}% ret</div>
                  </td>
                  <td style={{ padding: "10px 12px", color: C.textLo }}>{o.category}</td>
                  <td style={{ padding: "10px 12px", textAlign: "right", ...mono, color: C.textMid }}>{fmtINR(o.order_value)}</td>
                  <td style={{ padding: "10px 12px" }}>
                    <span style={{
                      ...mono, fontSize: 10, fontWeight: 500, padding: "2px 6px", borderRadius: 4,
                      background: o.is_cod ? C.amberBg : C.tealBg, color: o.is_cod ? C.amber : C.teal,
                    }}>{o.is_cod ? "COD" : "PREPAID"}</span>
                  </td>
                  <td style={{ padding: "10px 12px" }}>
                    <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                      <div style={{ width: 64, height: 5, borderRadius: 3, background: C.hair, overflow: "hidden" }}>
                        <div style={{ height: "100%", borderRadius: 3, width: `${o.risk_score}%`, background: tone.hex }} />
                      </div>
                      <span style={{ ...mono, fontSize: 11.5, fontWeight: 600, color: tone.hex }}>{o.risk_score}</span>
                    </div>
                  </td>
                  <td style={{ padding: "10px 12px" }}><StatusPill status={o.status} /></td>
                  <td style={{ padding: "10px 16px", textAlign: "right" }}>
                    {o.status === "pending" ? (
                      <div style={{ display: "flex", justifyContent: "flex-end", gap: 6 }}>
                        <button onClick={() => act(o.order_id, "verified")} className="mrs-action-btn" style={{
                          display: "flex", alignItems: "center", gap: 4, padding: "4px 8px", borderRadius: 6,
                          border: `1px solid #274A3A`, color: C.teal, fontSize: 10.5, fontWeight: 500, background: "transparent", cursor: "pointer",
                        }}>
                          <CheckCircle2 size={11} /> Verify
                        </button>
                        <button onClick={() => act(o.order_id, "blocked")} className="mrs-action-btn" style={{
                          display: "flex", alignItems: "center", gap: 4, padding: "4px 8px", borderRadius: 6,
                          border: `1px solid #4A2A28`, color: C.coral, fontSize: 10.5, fontWeight: 500, background: "transparent", cursor: "pointer",
                        }}>
                          <Ban size={11} /> Block COD
                        </button>
                      </div>
                    ) : (
                      <div style={{ fontSize: 10.5, color: C.textFaint }}>handled</div>
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
    </>
  );
}

function StatusPill({ status }) {
  const map = {
    pending: { c: C.amber, bg: C.amberBg, label: "Pending" },
    verified: { c: C.teal, bg: C.tealBg, label: "Verified" },
    blocked: { c: C.coral, bg: C.coralBg, label: "COD blocked" },
    letter_ready: { c: C.violet, bg: C.violetBg, label: "Letter ready" },
    submitted: { c: C.teal, bg: C.tealBg, label: "Submitted" },
  };
  const s = map[status] || map.pending;
  return <span style={{ fontSize: 10.5, fontWeight: 500, padding: "3px 8px", borderRadius: 999, color: s.c, background: s.bg }}>{s.label}</span>;
}

/* ============================================================================
   Disputes tab — kanban style
============================================================================ */

function DisputeCard({ d, onGenerate, onOpen }) {
  const winColor = d.win_probability >= 65 ? C.teal : d.win_probability >= 35 ? C.amber : C.coral;
  return (
    <div className="mrs-card" style={{ borderRadius: 10, border: `1px solid ${C.border}`, background: C.panel, padding: 14 }}>
      <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between" }}>
        <div>
          <div style={{ ...mono, fontSize: 11.5, color: C.textMid }}>
            {d.dispute_id}
            {d.live && <span style={{ marginLeft: 6, fontSize: 8.5, fontWeight: 700, color: C.violet, background: C.violetBg, padding: "1px 5px", borderRadius: 4 }}>LIVE</span>}
          </div>
          <div style={{ fontSize: 10.5, color: C.textFaint, marginTop: 2 }}>{d.order_id} · {fmtINR(d.order_value)}</div>
        </div>
        <RadialGauge value={d.win_probability} size={40} stroke={4} color={winColor} />
      </div>
      <div style={{ fontSize: 12, color: C.textMid, marginTop: 8 }}>{d.reason}</div>
      <div style={{ display: "flex", alignItems: "center", gap: 12, marginTop: 10, color: C.textFaint }}>
        <Truck size={11} color={d.has_signed_pod ? C.teal : "#3A4050"} />
        <Wifi size={11} color={d.ip_match ? C.teal : "#3A4050"} />
        <span style={{ display: "flex", alignItems: "center", gap: 3, fontSize: 10.5 }}>
          <MessageSquare size={11} color={d.comm_logs > 0 ? C.teal : "#3A4050"} />{d.comm_logs}
        </span>
      </div>
      <div style={{ marginTop: 12 }}>
        {d.status === "pending" && (
          <button onClick={() => onGenerate(d.dispute_id)} className="mrs-primary-btn" style={{
            width: "100%", display: "flex", alignItems: "center", justifyContent: "center", gap: 6,
            padding: "7px 0", borderRadius: 6, background: C.violet, color: "#12151C", fontSize: 11,
            fontWeight: 600, border: "none", cursor: "pointer",
          }}>
            <FileText size={12} /> Generate letter
          </button>
        )}
        {d.status === "letter_ready" && (
          <button onClick={() => onOpen(d.dispute_id)} className="mrs-action-btn" style={{
            width: "100%", display: "flex", alignItems: "center", justifyContent: "center", gap: 6,
            padding: "7px 0", borderRadius: 6, border: `1px solid ${C.borderHi}`, color: C.textMid, fontSize: 11,
            fontWeight: 500, background: "transparent", cursor: "pointer",
          }}>
            <FileText size={12} /> View & submit
          </button>
        )}
        {d.status === "submitted" && (
          <div style={{
            width: "100%", display: "flex", alignItems: "center", justifyContent: "center", gap: 6,
            padding: "7px 0", borderRadius: 6, background: C.tealBg, color: C.teal, fontSize: 11, fontWeight: 500,
          }}>
            <Clock size={12} /> Awaiting bank
          </div>
        )}
      </div>
    </div>
  );
}

function DisputesTab({ disputes, setDisputes, onGenerate, onOpen }) {
  const cols = [
    { key: "pending", title: "Pending", tone: C.amber },
    { key: "letter_ready", title: "Letter ready", tone: C.violet },
    { key: "submitted", title: "Submitted", tone: C.teal },
  ];
  return (
    <>
    <LiveDisputeScorer onAdd={(d) => setDisputes((prev) => [d, ...prev])} />
    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 16 }}>
      {cols.map((c) => {
        const items = disputes.filter((d) => d.status === c.key);
        return (
          <div key={c.key}>
            <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 12, padding: "0 4px" }}>
              <span style={{ width: 6, height: 6, borderRadius: "50%", background: c.tone }} />
              <span style={{ fontSize: 11.5, fontWeight: 500, color: C.textMid }}>{c.title}</span>
              <span style={{ ...mono, fontSize: 10.5, color: C.textFaint }}>{items.length}</span>
            </div>
            <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
              {items.map((d) => <DisputeCard key={d.dispute_id} d={d} onGenerate={onGenerate} onOpen={onOpen} />)}
              {items.length === 0 && (
                <div style={{ fontSize: 11, color: "#3A4050", padding: "24px 4px", textAlign: "center", border: `1px dashed ${C.border}`, borderRadius: 10 }}>none</div>
              )}
            </div>
          </div>
        );
      })}
    </div>
    </>
  );
}

/* ============================================================================
   Letter modal
============================================================================ */

function LetterModal({ dispute, onClose, onSubmit }) {
  const winColor = dispute.win_probability >= 65 ? C.teal : dispute.win_probability >= 35 ? C.amber : C.coral;
  return (
    <div style={{ position: "fixed", inset: 0, background: "rgba(0,0,0,0.6)", backdropFilter: "blur(4px)", display: "flex", alignItems: "center", justifyContent: "center", zIndex: 50, padding: "0 24px" }}>
      <div className="mrs-modal" style={{ background: C.panel, border: `1px solid ${C.border}`, borderRadius: 12, width: "100%", maxWidth: 540, maxHeight: "82vh", display: "flex", flexDirection: "column", boxShadow: "0 25px 50px rgba(0,0,0,0.5)" }}>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", padding: "14px 20px", borderBottom: `1px solid ${C.hair}` }}>
          <div>
            <div style={{ fontSize: 13, fontWeight: 600, color: C.textHi }}>Evidence letter</div>
            <div style={{ ...mono, fontSize: 10.5, color: C.textFaint }}>{dispute.dispute_id} · {dispute.order_id}</div>
          </div>
          <button onClick={onClose} style={{ background: "none", border: "none", color: C.textFaint, cursor: "pointer", padding: 4 }}><X size={17} /></button>
        </div>
        <div style={{ padding: "10px 20px", display: "flex", alignItems: "center", gap: 8, borderBottom: `1px solid ${C.hair}`, background: C.sunken }}>
          <span style={{ fontSize: 10.5, color: "#7C8598" }}>Win probability</span>
          <span style={{ ...mono, fontSize: 12.5, fontWeight: 600, color: winColor }}>{dispute.win_probability}%</span>
          <span style={{ fontSize: 10.5, color: C.textFaint }}>— Random Forest, {DATASET_SIZES.disputes.toLocaleString("en-IN")}-case model</span>
        </div>
        <div style={{ padding: 20, overflow: "auto", flex: 1 }}>
          <pre style={{ whiteSpace: "pre-wrap", ...mono, fontSize: 11.5, lineHeight: 1.6, color: C.textMid, background: C.sunken, border: `1px solid ${C.hair}`, borderRadius: 8, padding: 16, margin: 0 }}>{dispute.letter}</pre>
        </div>
        <div style={{ padding: "14px 20px", borderTop: `1px solid ${C.hair}`, display: "flex", justifyContent: "flex-end", gap: 8 }}>
          <button onClick={onClose} className="mrs-action-btn" style={{ padding: "7px 14px", borderRadius: 8, border: `1px solid ${C.border}`, color: C.textLo, fontSize: 11.5, fontWeight: 500, background: "transparent", cursor: "pointer" }}>Close</button>
          <button onClick={onSubmit} className="mrs-primary-btn" style={{ display: "flex", alignItems: "center", gap: 6, padding: "7px 14px", borderRadius: 8, background: C.violet, color: "#12151C", fontSize: 11.5, fontWeight: 600, border: "none", cursor: "pointer" }}>
            <Send size={12} /> Submit response
          </button>
        </div>
      </div>
    </div>
  );
}

/* ============================================================================
   Model Lab tab
============================================================================ */

function ModelPanel({ m, color }) {
  const cm = m.confusion;
  return (
    <div style={{ borderRadius: 12, border: `1px solid ${C.border}`, background: C.panel, padding: 16 }}>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
        <div>
          <div style={{ fontSize: 13, fontWeight: 600, color: C.textHi }}>{m.name}</div>
          <div style={{ ...mono, fontSize: 10.5, color: C.textFaint, marginTop: 2 }}>{m.algo} · {m.detail}</div>
        </div>
        <Cpu size={16} color={color} />
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr 1fr", gap: 8, marginTop: 16 }}>
        {Object.entries(m.metrics).map(([k, v]) => (
          <div key={k} style={{ borderRadius: 8, border: `1px solid ${C.hair}`, background: C.sunken, padding: 10, textAlign: "center" }}>
            <div style={{ ...mono, fontSize: 16, fontWeight: 600, color }}>{v}</div>
            <div style={{ fontSize: 9.5, color: C.textFaint, textTransform: "uppercase", letterSpacing: "0.04em", marginTop: 2 }}>{k}</div>
          </div>
        ))}
      </div>

      <div style={{ marginTop: 16 }}>
        <div style={{ fontSize: 10.5, color: "#7C8598", marginBottom: 8 }}>Feature importance</div>
        <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
          {m.importances.map((f) => (
            <div key={f.f} style={{ display: "flex", alignItems: "center", gap: 8 }}>
              <div style={{ width: 128, fontSize: 10.5, ...mono, color: C.textLo, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{f.f}</div>
              <div style={{ flex: 1, height: 6, borderRadius: 3, background: C.hair, overflow: "hidden" }}>
                <div style={{ height: "100%", borderRadius: 3, width: `${(f.v / m.importances[0].v) * 100}%`, background: color }} />
              </div>
              <div style={{ width: 36, fontSize: 10, ...mono, color: C.textFaint, textAlign: "right" }}>{(f.v * 100).toFixed(1)}%</div>
            </div>
          ))}
        </div>
      </div>

      <div style={{ marginTop: 16 }}>
        <div style={{ fontSize: 10.5, color: "#7C8598", marginBottom: 8 }}>Confusion matrix (test split)</div>
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 4, width: 160 }}>
          {[["TN", cm[0][0]], ["FP", cm[0][1]], ["FN", cm[1][0]], ["TP", cm[1][1]]].map(([label, val]) => (
            <div key={label} style={{
              borderRadius: 6, padding: 8, textAlign: "center", border: `1px solid ${C.hair}`,
              background: label === "TP" || label === "TN" ? "rgba(52,216,176,0.08)" : "rgba(255,98,89,0.08)",
            }}>
              <div style={{ ...mono, fontSize: 13, color: C.textHi }}>{val}</div>
              <div style={{ fontSize: 9, color: C.textFaint }}>{label}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function MiniStat({ label, value }) {
  return (
    <div style={{ borderRadius: 8, border: `1px solid ${C.hair}`, background: C.sunken, padding: 10 }}>
      <div style={{ ...mono, fontSize: 15, color: C.textHi }}>{value}</div>
      <div style={{ fontSize: 10, color: C.textFaint, marginTop: 2 }}>{label}</div>
    </div>
  );
}

function ModelLabTab() {
  return (
    <div>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
        <ModelPanel m={MODEL1} color={C.coral} />
        <ModelPanel m={MODEL2} color={C.violet} />
      </div>
      <div style={{ borderRadius: 12, border: `1px solid ${C.border}`, background: C.panel, padding: 16, marginTop: 16 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 12 }}>
          <Database size={13} color={C.amber} />
          <div style={{ fontSize: 11.5, color: C.textMid, fontWeight: 500 }}>Training data</div>
        </div>
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 12 }}>
          <MiniStat label="Orders (synthetic)" value={DATASET_SIZES.orders.toLocaleString("en-IN")} />
          <MiniStat label="Disputes (synthetic)" value={DATASET_SIZES.disputes.toLocaleString("en-IN")} />
          <MiniStat label="Train / test split" value="75 / 25" />
        </div>
        <div style={{ fontSize: 10.5, color: C.textFaint, marginTop: 12, lineHeight: 1.6 }}>
          Both datasets are generated with a hand-tuned latent "risk proneness" signal per synthetic customer
          (generate_data.py), so the label isn't random — precision/recall above are honest sklearn/xgboost
          evaluation on a held-out 25% split, not hand-picked numbers.
        </div>
      </div>
    </div>
  );
}

function LoginScreen({ onLogin }) {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [remember, setRemember] = useState(true);
  const [error, setError] = useState("");

  function handleSubmit(e) {
    e.preventDefault();
    setError("");

    if (
      email.trim().toLowerCase() !== "demo@merchant.com" ||
      password !== "demo123"
    ) {
      setError("Invalid merchant credentials. Use the demo account shown below.");
      return;
    }

    onLogin({ remember });
  }

  const inputBase = {
    width: "100%",
    background: "#0B0E14",
    border: `1px solid ${C.border}`,
    borderRadius: 8,
    padding: "11px 12px",
    fontSize: 12,
    color: C.textHi,
    outline: "none",
    boxSizing: "border-box",
    transition: "border-color .15s, box-shadow .15s",
  };

  return (
    <div style={{
      minHeight: "100vh",
      width: "100%",
      background: C.bg,
      color: C.textHi,
      fontFamily: "'Inter', ui-sans-serif, system-ui",
      position: "relative",
      overflow: "hidden",
      display: "flex",
      flexDirection: "column",
    }}>
      <style>{`
        .mrs-login-input:focus {
          border-color: ${C.violet} !important;
          box-shadow: 0 0 0 3px ${C.violet}18 !important;
        }
        .mrs-login-btn:hover {
          transform: translateY(-1px);
          filter: brightness(1.05);
        }
        .mrs-login-btn:active {
          transform: translateY(0);
        }
        @media (max-width: 820px) {
          .mrs-login-grid {
            grid-template-columns: 1fr !important;
            max-width: 460px !important;
          }
          .mrs-login-hero {
            display: none !important;
          }
        }
      `}</style>

      {/* Ambient grid / glow — deliberately subtle to match the dashboard */}
      <div style={{
        position: "absolute",
        inset: 0,
        pointerEvents: "none",
        opacity: 0.35,
        backgroundImage: `
          linear-gradient(${C.border} 1px, transparent 1px),
          linear-gradient(90deg, ${C.border} 1px, transparent 1px)
        `,
        backgroundSize: "72px 72px",
        maskImage: "linear-gradient(to bottom, black, transparent 75%)",
      }} />
      <div style={{
        position: "absolute",
        width: 680,
        height: 680,
        borderRadius: "50%",
        top: -360,
        right: -250,
        background: `radial-gradient(circle, ${C.violet}15 0%, transparent 68%)`,
        pointerEvents: "none",
      }} />
      <div style={{
        position: "absolute",
        width: 560,
        height: 560,
        borderRadius: "50%",
        bottom: -340,
        left: -250,
        background: `radial-gradient(circle, ${C.amber}10 0%, transparent 68%)`,
        pointerEvents: "none",
      }} />

      {/* Minimal brand bar */}
      <div style={{
        height: 66,
        padding: "0 34px",
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        borderBottom: `1px solid ${C.hair}`,
        position: "relative",
        zIndex: 2,
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <div style={{
            width: 32,
            height: 32,
            borderRadius: 8,
            background: `linear-gradient(135deg, ${C.amber}, ${C.coral})`,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            boxShadow: `0 8px 24px ${C.amber}16`,
          }}>
            <Shield size={16} color={C.bg} />
          </div>
          <div>
            <div style={{ ...display, fontSize: 14.5, fontWeight: 600, lineHeight: 1.1 }}>
              Merchant Risk Shield
            </div>
            <div style={{ fontSize: 9.5, color: C.textFaint, marginTop: 3 }}>
              Intelligent merchant protection
            </div>
          </div>
        </div>

        <div style={{
          display: "flex",
          alignItems: "center",
          gap: 6,
          fontSize: 10,
          color: C.textFaint,
        }}>
          <span style={{
            width: 6,
            height: 6,
            borderRadius: "50%",
            background: C.teal,
            boxShadow: `0 0 10px ${C.teal}80`,
          }} />
          Systems operational
        </div>
      </div>

      <div className="mrs-login-grid" style={{
        width: "100%",
        maxWidth: 1050,
        margin: "auto",
        padding: "64px 34px",
        boxSizing: "border-box",
        display: "grid",
        gridTemplateColumns: "1fr 400px",
        gap: 90,
        alignItems: "center",
        position: "relative",
        zIndex: 1,
      }}>

        {/* Left side — product story */}
        <div className="mrs-login-hero">
          <div style={{
            display: "inline-flex",
            alignItems: "center",
            gap: 7,
            border: `1px solid ${C.border}`,
            background: C.panel,
            borderRadius: 999,
            padding: "5px 9px",
            color: C.textLo,
            fontSize: 9.5,
            marginBottom: 20,
          }}>
            <Sparkles size={11} color={C.amber} />
            AI risk intelligence for merchants
          </div>

          <h1 style={{
            ...display,
            fontSize: 46,
            lineHeight: 1.05,
            letterSpacing: "-0.045em",
            fontWeight: 600,
            maxWidth: 600,
            margin: "0 0 18px",
          }}>
            Protect every order.<br />
            <span style={{ color: C.violet }}>Recover every winnable dispute.</span>
          </h1>

          <p style={{
            maxWidth: 560,
            color: C.textLo,
            fontSize: 13.5,
            lineHeight: 1.75,
            margin: 0,
          }}>
            Merchant Risk Shield combines real-time order risk scoring,
            chargeback outcome prediction, and evidence generation into one
            merchant intelligence workspace.
          </p>

          <div style={{
            display: "grid",
            gridTemplateColumns: "repeat(3, 1fr)",
            maxWidth: 570,
            marginTop: 34,
            borderTop: `1px solid ${C.hair}`,
            borderBottom: `1px solid ${C.hair}`,
          }}>
            {[
              ["6,000", "orders analyzed"],
              ["1,400", "disputes modeled"],
              ["2", "ML engines"],
            ].map(([value, label], i) => (
              <div key={label} style={{
                padding: "16px 14px 16px 0",
                borderRight: i < 2 ? `1px solid ${C.hair}` : "none",
                marginRight: i < 2 ? 14 : 0,
              }}>
                <div style={{ ...mono, fontSize: 19, fontWeight: 600, color: C.textHi }}>
                  {value}
                </div>
                <div style={{ fontSize: 9.5, color: C.textFaint, marginTop: 4 }}>
                  {label}
                </div>
              </div>
            ))}
          </div>

          <div style={{
            display: "flex",
            alignItems: "center",
            gap: 9,
            marginTop: 25,
            color: C.textFaint,
            fontSize: 10,
          }}>
            <CheckCircle2 size={12} color={C.teal} />
            Evidence-backed decisions, not black-box actions
          </div>
        </div>

        {/* Login card */}
        <form onSubmit={handleSubmit} style={{
          background: "rgba(18,21,28,0.94)",
          border: `1px solid ${C.border}`,
          borderRadius: 14,
          padding: 26,
          boxShadow: "0 30px 80px rgba(0,0,0,0.42)",
          backdropFilter: "blur(16px)",
        }}>
          <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 22 }}>
            <div>
              <div style={{ ...display, fontSize: 19, fontWeight: 600 }}>
                Sign in
              </div>
              <div style={{ fontSize: 10.5, color: C.textFaint, marginTop: 4 }}>
                Access your merchant control center
              </div>
            </div>
            <div style={{
              width: 34,
              height: 34,
              borderRadius: 9,
              background: C.violetBg,
              border: `1px solid ${C.violet}28`,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              color: C.violet,
            }}>
              <Lock size={15} />
            </div>
          </div>

          <label style={{ display: "block", fontSize: 10, color: C.textLo, marginBottom: 6 }}>
            MERCHANT EMAIL
          </label>
          <input
            className="mrs-login-input"
            type="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            placeholder="merchant@company.com"
            autoComplete="email"
            style={{ ...inputBase, marginBottom: 15 }}
          />

          <label style={{ display: "block", fontSize: 10, color: C.textLo, marginBottom: 6 }}>
            PASSWORD
          </label>
          <div style={{ position: "relative" }}>
            <input
              className="mrs-login-input"
              type={showPassword ? "text" : "password"}
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="Enter your password"
              autoComplete="current-password"
              style={{ ...inputBase, paddingRight: 42 }}
            />
            <button
              type="button"
              onClick={() => setShowPassword((v) => !v)}
              style={{
                position: "absolute",
                right: 9,
                top: "50%",
                transform: "translateY(-50%)",
                border: "none",
                background: "transparent",
                color: C.textFaint,
                cursor: "pointer",
                padding: 5,
              }}
            >
              {showPassword ? <EyeOff size={14} /> : <Eye size={14} />}
            </button>
          </div>

          <div style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            marginTop: 13,
          }}>
            <label style={{
              display: "flex",
              alignItems: "center",
              gap: 7,
              color: C.textFaint,
              fontSize: 10,
              cursor: "pointer",
            }}>
              <input
                type="checkbox"
                checked={remember}
                onChange={(e) => setRemember(e.target.checked)}
              />
              Keep me signed in
            </label>
            <span style={{ color: C.textFaint, fontSize: 9.5 }}>
              Secure session
            </span>
          </div>

          {error && (
            <div style={{
              marginTop: 14,
              padding: "9px 10px",
              borderRadius: 7,
              background: C.coralBg,
              border: "1px solid #4A2A28",
              color: C.coral,
              fontSize: 10,
              lineHeight: 1.45,
            }}>
              {error}
            </div>
          )}

          <button
            type="submit"
            className="mrs-login-btn"
            style={{
              width: "100%",
              marginTop: 20,
              padding: "11px 14px",
              borderRadius: 8,
              background: `linear-gradient(135deg, ${C.violet}, #7F70E8)`,
              color: "#0D0F14",
              fontSize: 11.5,
              fontWeight: 700,
              border: "none",
              cursor: "pointer",
              transition: "transform .15s, filter .15s",
              boxShadow: `0 10px 28px ${C.violet}20`,
            }}
          >
            Continue to dashboard
            <ArrowUpRight size={13} style={{ verticalAlign: "middle", marginLeft: 5 }} />
          </button>

          <div style={{
            marginTop: 18,
            paddingTop: 15,
            borderTop: `1px solid ${C.hair}`,
          }}>
            <div style={{
              display: "flex",
              alignItems: "center",
              gap: 6,
              color: C.textFaint,
              fontSize: 9,
              letterSpacing: "0.04em",
              marginBottom: 9,
            }}>
              <Zap size={10} color={C.amber} />
              BUILDATHON DEMO ACCOUNT
            </div>

            <div style={{
              display: "grid",
              gridTemplateColumns: "1fr 1fr",
              gap: 7,
            }}>
              <div style={{
                background: C.sunken,
                border: `1px solid ${C.hair}`,
                borderRadius: 6,
                padding: "7px 8px",
              }}>
                <div style={{ fontSize: 8.5, color: C.textFaint }}>EMAIL</div>
                <div style={{ ...mono, fontSize: 9.5, color: C.textMid, marginTop: 3 }}>
                  demo@merchant.com
                </div>
              </div>
              <div style={{
                background: C.sunken,
                border: `1px solid ${C.hair}`,
                borderRadius: 6,
                padding: "7px 8px",
              }}>
                <div style={{ fontSize: 8.5, color: C.textFaint }}>PASSWORD</div>
                <div style={{ ...mono, fontSize: 9.5, color: C.textMid, marginTop: 3 }}>
                  demo123
                </div>
              </div>
            </div>
          </div>

          <div style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            gap: 6,
            marginTop: 17,
            color: C.textFaint,
            fontSize: 9,
          }}>
            <Shield size={10} />
            Merchant workspace · Protected access
          </div>
        </form>
      </div>

      <div style={{
        padding: "13px 34px",
        borderTop: `1px solid ${C.hair}`,
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        color: C.textFaint,
        fontSize: 9,
        position: "relative",
        zIndex: 2,
      }}>
        <span>Merchant Risk Shield · AI dispute & risk intelligence</span>
        <span style={{ ...mono }}>v1.0 · demo environment</span>
      </div>
    </div>
  );
}


/* ============================================================================
   App shell
============================================================================ */

export default function MerchantRiskShield() {
  const [authenticated, setAuthenticated] = useState(() => {
    try {
      return sessionStorage.getItem("mrs_authenticated") === "true";
    } catch {
      return false;
    }
  });
  const [tab, setTab] = useState("orders");
  const [orders, setOrders] = useState(RAW_ORDERS);
  const [disputes, setDisputes] = useState(RAW_DISPUTES);
  const [letterId, setLetterId] = useState(null);
  const [toast, setToast] = useState(null);

  function notify(msg) { setToast(msg); setTimeout(() => setToast(null), 2200); }

  function handleLogin({ remember }) {
    setAuthenticated(true);
    try {
      if (remember) sessionStorage.setItem("mrs_authenticated", "true");
    } catch {}
  }

  function handleLogout() {
    try {
      sessionStorage.removeItem("mrs_authenticated");
    } catch {}
    setAuthenticated(false);
    setTab("orders");
    setLetterId(null);
  }

  if (!authenticated) {
    return <LoginScreen onLogin={handleLogin} />;
  }

  async function generateLetter(id) {
    try {
      const res = await fetch(`https://merchant-risk-shield-api.onrender.com/api/disputes/${id}/generate-letter`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
      });

      if (!res.ok) throw new Error(`Letter generation failed: ${res.status}`);

      const updatedDispute = await res.json();

      setDisputes((prev) =>
        prev.map((d) => d.dispute_id === id ? updatedDispute : d)
      );

      setLetterId(id);
      notify(`Evidence letter generated for ${id}`);
    } catch (error) {
      console.error("Letter generation failed:", error);
      notify("Failed to generate evidence letter");
    }
  }
  function submitResponse(id) {
    setDisputes((prev) => prev.map((d) => d.dispute_id === id ? { ...d, status: "submitted" } : d));
    setLetterId(null);
    notify(`Evidence submitted for ${id}`);
  }

  const active = disputes.find((d) => d.dispute_id === letterId);

  return (
    <div style={{ fontFamily: "'Inter', ui-sans-serif, system-ui", width: "100%", minHeight: 760, background: C.bg, color: C.textHi, padding: 20, boxSizing: "border-box" }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap');
        .mrs-row:hover { background: ${C.panelAlt}; }
        .mrs-filter-btn:hover { color: ${C.textHi} !important; }
        .mrs-action-btn:hover { border-color: ${C.borderHi} !important; }
        .mrs-primary-btn:hover { background: #B0A3FF !important; }
        .mrs-card:hover { border-color: ${C.borderHi}; }
        .mrs-pulse-wrap { position: relative; display: inline-flex; height: 8px; width: 8px; }
        .mrs-pulse-ping { animation: mrsPing 1.6s cubic-bezier(0,0,0.2,1) infinite; position: absolute; inline-size: 100%; block-size: 100%; border-radius: 9999px; background: ${C.teal}; opacity: 0.6; }
        .mrs-pulse-dot { position: relative; display: inline-flex; border-radius: 9999px; height: 8px; width: 8px; background: ${C.teal}; }
        @keyframes mrsPing { 75%, 100% { transform: scale(2.2); opacity: 0; } }
        .mrs-spin { animation: mrsSpin 0.8s linear infinite; }
        @keyframes mrsSpin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
        .mrs-modal { animation: mrsSlideUp .25s ease-out; }
        @keyframes mrsSlideUp { from { opacity: 0; transform: translateY(12px); } to { opacity: 1; transform: translateY(0); } }
        .mrs-toast { animation: mrsFadeIn .2s ease-out; }
        @keyframes mrsFadeIn { from { opacity: 0; transform: translate(-50%, 8px); } to { opacity: 1; transform: translate(-50%, 0); } }
      `}</style>

      {/* Top bar */}
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 20, flexWrap: "wrap", gap: 12 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <div style={{ width: 32, height: 32, borderRadius: 8, background: `linear-gradient(135deg, ${C.amber}, ${C.coral})`, display: "flex", alignItems: "center", justifyContent: "center" }}>
            <Shield size={16} color={C.bg} />
          </div>
          <div>
            <div style={{ ...display, fontSize: 15, fontWeight: 600, lineHeight: 1.2, color: C.textHi }}>Merchant Risk Shield</div>
            <div style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 10.5, color: C.textFaint }}>
              <Pulse /> engine live · scoring in real time
            </div>
          </div>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 4, background: C.panel, border: `1px solid ${C.border}`, borderRadius: 10, padding: 4 }}>
          {[
            { k: "orders", label: "Orders", icon: <Layers size={13} /> },
            { k: "disputes", label: "Disputes", icon: <Scale size={13} /> },
            { k: "lab", label: "Model Lab", icon: <Cpu size={13} /> },
          ].map((t) => (
            <button key={t.k} onClick={() => setTab(t.k)} style={{
              display: "flex", alignItems: "center", gap: 6, padding: "7px 12px", borderRadius: 8, fontSize: 12,
              fontWeight: 500, cursor: "pointer", border: "none",
              background: tab === t.k ? "#1E232E" : "transparent",
              color: tab === t.k ? C.textHi : "#7C8598",
            }}>
              {t.icon}{t.label}
            </button>
          ))}
        </div>
        <button
          onClick={handleLogout}
          className="mrs-action-btn"
          title="Sign out"
          style={{
            display: "flex", alignItems: "center", gap: 6,
            padding: "7px 10px", borderRadius: 8,
            border: `1px solid ${C.border}`, color: C.textLo,
            fontSize: 11, fontWeight: 500, background: C.panel,
            cursor: "pointer",
          }}
        >
          <LogOut size={12} /> Sign out
        </button>
      </div>

      <Overview orders={orders} disputes={disputes} />

      {tab === "orders" && <OrdersTab orders={orders} setOrders={setOrders} />}
      {tab === "disputes" && <DisputesTab disputes={disputes} setDisputes={setDisputes} onGenerate={generateLetter} onOpen={(id) => setLetterId(id)} />}
      {tab === "lab" && <ModelLabTab />}

      {active && <LetterModal dispute={active} onClose={() => setLetterId(null)} onSubmit={() => submitResponse(active.dispute_id)} />}

      {toast && (
        <div className="mrs-toast" style={{
          position: "fixed", bottom: 24, left: "50%", transform: "translateX(-50%)", background: C.panel,
          border: `1px solid ${C.border}`, color: C.textHi, fontSize: 12, padding: "10px 16px", borderRadius: 10,
          boxShadow: "0 20px 40px rgba(0,0,0,0.5)", display: "flex", alignItems: "center", gap: 8, zIndex: 50,
        }}>
          <CheckCircle2 size={13} color={C.teal} /> {toast}
        </div>
      )}
    </div>
  );
}

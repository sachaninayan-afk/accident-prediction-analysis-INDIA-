# -*- coding: utf-8 -*-
"""
================================================================================
  ALL_CODE.py  —  India Road Accident Severity Predictor
  -------------------------------------------------------
  Single file containing:
    SECTION 1 : ML Model Training   (Random Forest)
    SECTION 2 : Flask REST API      (Backend  :5000)
    SECTION 3 : Streamlit UI        (Frontend :8501)

  HOW TO RUN:
  -----------
    python ALL_CODE.py            ->  starts Flask backend + Streamlit frontend
    python ALL_CODE.py train      ->  only train/retrain the model
    python ALL_CODE.py serve      ->  only start the Flask backend
    streamlit run ALL_CODE.py     ->  only start the Streamlit frontend

================================================================================
"""

import os
import sys
import pickle
import threading
import subprocess
import requests as _req
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from flask import Flask, request, jsonify
from flask_cors import CORS
try:
    import streamlit as st
    _ST_AVAILABLE = True
except ImportError:
    _ST_AVAILABLE = False

# ── Shared paths ───────────────────────────────────────────────────────────────
_HERE      = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(_HERE, "data", "accident_prediction_india.csv")
MODEL_PATH = os.path.join(_HERE, "model", "accident_model.pkl")
ENC_PATH   = os.path.join(_HERE, "model", "encoders.pkl")

# ── Shared lookup maps ─────────────────────────────────────────────────────────
MONTH_MAP = {
    "January": 1, "February": 2, "March": 3,  "April": 4,
    "May": 5,     "June": 6,     "July": 7,    "August": 8,
    "September": 9,"October": 10,"November": 11,"December": 12,
}
DAY_MAP = {
    "Monday": 1, "Tuesday": 2, "Wednesday": 3, "Thursday": 4,
    "Friday": 5, "Saturday": 6, "Sunday": 7,
}


# ==============================================================================
# SECTION 1 — ML MODEL TRAINING
# ==============================================================================
def run_training():
    """Train the Random Forest model and save artefacts to model/ folder."""
    df = pd.read_csv(DATA_PATH)
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

    df["Hour"] = df["Time of Day"].apply(
        lambda t: int(str(t).split(":")[0]) if pd.notna(t) else 12
    )
    df["Month_Num"] = df["Month"].map(MONTH_MAP).fillna(6).astype(int)
    df["Day_Num"]   = df["Day of Week"].map(DAY_MAP).fillna(1).astype(int)

    CATEGORICAL = [
        "Weather Conditions", "Road Type", "Road Condition",
        "Lighting Conditions", "Traffic Control Presence",
        "Vehicle Type Involved", "Driver Gender",
        "Driver License Status", "Alcohol Involvement",
        "Accident Location Details",
    ]
    NUMERICAL = [
        "Number of Vehicles Involved", "Number of Casualties",
        "Number of Fatalities", "Speed Limit (km/h)", "Driver Age",
        "Hour", "Month_Num", "Day_Num",
    ]
    TARGET = "Accident Severity"

    df = df.dropna(subset=[TARGET])

    encoders = {}
    for col in CATEGORICAL:
        le = LabelEncoder()
        df[col + "_enc"] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    target_le = LabelEncoder()
    df["Target"] = target_le.fit_transform(df[TARGET])
    encoders["__target__"] = target_le

    feature_cols = NUMERICAL + [c + "_enc" for c in CATEGORICAL]
    X = df[feature_cols].fillna(0).astype(float)
    y = df["Target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200, max_depth=15,
        min_samples_split=5, random_state=42, n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(f"\nTest Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred, target_names=target_le.classes_))

    meta = {"feature_cols": feature_cols, "categorical": CATEGORICAL, "numerical": NUMERICAL}
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, "wb") as f: pickle.dump(model, f)
    with open(ENC_PATH,   "wb") as f: pickle.dump({"encoders": encoders, "meta": meta}, f)

    print(f"Model saved    -> {MODEL_PATH}")
    print(f"Encoders saved -> {ENC_PATH}")


# ==============================================================================
# SECTION 2 — FLASK REST API BACKEND
# ==============================================================================
OPTIONS = {
    "Weather Conditions":        ["Clear", "Rainy", "Foggy", "Hazy", "Stormy"],
    "Road Type":                 ["National Highway", "State Highway", "Urban Road", "Village Road"],
    "Road Condition":            ["Dry", "Wet", "Damaged", "Under Construction"],
    "Lighting Conditions":       ["Daylight", "Dark", "Dawn", "Dusk"],
    "Traffic Control Presence":  ["Signs", "Signals", "Police Checkpost", "None"],
    "Vehicle Type Involved":     ["Car", "Truck", "Bus", "Two-Wheeler", "Cycle", "Auto-Rickshaw", "Pedestrian"],
    "Driver Gender":             ["Male", "Female"],
    "Driver License Status":     ["Valid", "Expired", "None"],
    "Alcohol Involvement":       ["Yes", "No"],
    "Accident Location Details": ["Straight Road", "Curve", "Intersection", "Bridge"],
}


def _load_bundle():
    with open(MODEL_PATH, "rb") as f: model = pickle.load(f)
    with open(ENC_PATH,   "rb") as f: bundle = pickle.load(f)
    return model, bundle["encoders"], bundle["meta"]


def _build_vector(data, encoders, feature_cols, categorical):
    row = {
        "Number of Vehicles Involved": float(data.get("Number of Vehicles Involved", 2)),
        "Number of Casualties":        float(data.get("Number of Casualties", 0)),
        "Number of Fatalities":        float(data.get("Number of Fatalities", 0)),
        "Speed Limit (km/h)":          float(data.get("Speed Limit (km/h)", 60)),
        "Driver Age":                  float(data.get("Driver Age", 30)),
        "Hour":                        float(data.get("Hour", 12)),
        "Month_Num":                   float(MONTH_MAP.get(data.get("Month", "June"), 6)),
        "Day_Num":                     float(DAY_MAP.get(data.get("Day of Week", "Monday"), 1)),
    }
    for col in categorical:
        le  = encoders[col]
        val = str(data.get(col, le.classes_[0]))
        if val not in le.classes_: val = le.classes_[0]
        row[col + "_enc"] = float(le.transform([val])[0])
    return pd.DataFrame([[row[fc] for fc in feature_cols]], columns=feature_cols)


def create_flask_app():
    model, encoders, meta = _load_bundle()
    target_le    = encoders["__target__"]
    feature_cols = meta["feature_cols"]
    categorical  = meta["categorical"]

    app = Flask(__name__)
    CORS(app)

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({"status": "ok", "model": "RandomForestClassifier",
                        "classes": list(target_le.classes_)})

    @app.route("/options", methods=["GET"])
    def options():
        return jsonify({"categorical_options": OPTIONS,
                        "months": list(MONTH_MAP.keys()),
                        "days":   list(DAY_MAP.keys())})

    @app.route("/predict", methods=["POST"])
    def predict():
        payload = request.get_json(force=True)
        if not payload:
            return jsonify({"error": "Empty JSON body"}), 400
        try:
            X       = _build_vector(payload, encoders, feature_cols, categorical)
            pred_id = model.predict(X)[0]
            proba   = model.predict_proba(X)[0]
            label   = target_le.inverse_transform([pred_id])[0]
            probs   = {c: round(float(p)*100, 2) for c, p in zip(target_le.classes_, proba)}
            colour  = {"Minor": "green", "Serious": "orange", "Fatal": "red"}.get(label, "grey")
            return jsonify({"prediction": label, "confidence": round(float(max(proba))*100, 2),
                            "probabilities": probs, "colour": colour})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/feature_importance", methods=["GET"])
    def feature_importance():
        n   = int(request.args.get("top", 10))
        imp = model.feature_importances_
        idx = np.argsort(imp)[::-1][:n]
        return jsonify([{"feature": feature_cols[i], "importance": round(float(imp[i]), 4)}
                        for i in idx])

    return app


def run_flask_server():
    app = create_flask_app()
    print("=" * 50)
    print("  Flask backend  ->  http://127.0.0.1:5000")
    print("=" * 50)
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)


# ==============================================================================
# SECTION 3 — STREAMLIT FRONTEND
# Detected via environment variable set by _start_streamlit_subprocess()
# ==============================================================================
_RUNNING_IN_STREAMLIT = os.environ.get("STREAMLIT_LAUNCHED_FROM_ALL_CODE") == "1"

if _RUNNING_IN_STREAMLIT:
    API_BASE  = "http://127.0.0.1:5000"
    ST_DATA   = os.path.join(_HERE, "data", "accident_prediction_india.csv")

    st.set_page_config(page_title="India Accident Severity Predictor",
                       page_icon="🚦", layout="wide")

    st.markdown("""
    <style>
        .main-header{font-size:2.2rem;font-weight:700;color:#1f2328;margin-bottom:.2rem}
        .sub-header{font-size:1rem;color:#57606a;margin-bottom:1.5rem}
        .result-box{padding:1.4rem 1.8rem;border-radius:10px;border-left:6px solid;margin-top:1rem}
        .result-minor{border-color:#22c55e;background:#f0fdf4}
        .result-serious{border-color:#f97316;background:#fff7ed}
        .result-fatal{border-color:#ef4444;background:#fef2f2}
        .metric-label{font-size:.85rem;color:#57606a;font-weight:600}
        .metric-value{font-size:1.8rem;font-weight:700}
        .stButton>button{background:#3b82d4;color:white;border:none;border-radius:6px;
                         padding:.5rem 2rem;font-size:1rem;font-weight:600}
        .stButton>button:hover{background:#2563eb}
    </style>""", unsafe_allow_html=True)

    st.markdown('<div class="main-header">🚦 India Road Accident Severity Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Enter accident scenario details to predict severity — Minor, Serious, or Fatal</div>', unsafe_allow_html=True)
    st.divider()

    @st.cache_data(show_spinner=False)
    def fetch_options():
        try:    return _req.get(f"{API_BASE}/options", timeout=5).json()
        except: return None

    @st.cache_data(show_spinner=False)
    def fetch_importance():
        try:    return _req.get(f"{API_BASE}/feature_importance?top=12", timeout=5).json()
        except: return []

    @st.cache_data(show_spinner=False)
    def load_data():
        try:    return pd.read_csv(ST_DATA)
        except: return pd.DataFrame()

    opts = fetch_options()
    if opts is None:
        st.error("Flask backend not running. Open a terminal and run:  python ALL_CODE.py serve")
        st.stop()

    cat_opts = opts["categorical_options"]
    months   = opts["months"]
    days     = opts["days"]

    tab1, tab2, tab3, tab4 = st.tabs(["🔮 Predict", "📊 Feature Importance", "📂 Dataset Explorer", "ℹ️ About"])

    # ── TAB 1 : PREDICT ───────────────────────────────────────────────────────
    with tab1:
        c_form, c_result = st.columns([1.1, 0.9], gap="large")
        with c_form:
            st.subheader("Accident Scenario Details")
            with st.form("form"):
                a, b = st.columns(2)
                with a:
                    weather      = st.selectbox("Weather Conditions",       cat_opts["Weather Conditions"])
                    road_type    = st.selectbox("Road Type",                cat_opts["Road Type"])
                    road_cond    = st.selectbox("Road Condition",           cat_opts["Road Condition"])
                    lighting     = st.selectbox("Lighting Conditions",      cat_opts["Lighting Conditions"])
                    traffic_ctrl = st.selectbox("Traffic Control Presence", cat_opts["Traffic Control Presence"])
                with b:
                    vehicle_type   = st.selectbox("Vehicle Type Involved",  cat_opts["Vehicle Type Involved"])
                    gender         = st.selectbox("Driver Gender",          cat_opts["Driver Gender"])
                    license_status = st.selectbox("Driver License Status",  cat_opts["Driver License Status"])
                    alcohol        = st.selectbox("Alcohol Involvement",    cat_opts["Alcohol Involvement"])
                    location_det   = st.selectbox("Accident Location",      cat_opts["Accident Location Details"])
                st.markdown("---")
                n1, n2, n3 = st.columns(3)
                with n1:
                    num_vehicles   = st.slider("Number of Vehicles",   1, 10, 2)
                    num_casualties = st.slider("Number of Casualties", 0, 20, 2)
                    num_fatalities = st.slider("Number of Fatalities", 0, 10, 0)
                with n2:
                    speed_limit = st.slider("Speed Limit (km/h)", 20, 140, 60)
                    driver_age  = st.slider("Driver Age",         16, 80,  30)
                    hour        = st.slider("Hour of Day",        0,  23,  12)
                with n3:
                    month       = st.selectbox("Month", months)
                    day_of_week = st.selectbox("Day",   days)
                submitted = st.form_submit_button("🔮  Predict Severity", use_container_width=True)

        with c_result:
            st.subheader("Prediction Result")
            if submitted:
                payload = {
                    "Weather Conditions": weather, "Road Type": road_type,
                    "Road Condition": road_cond, "Lighting Conditions": lighting,
                    "Traffic Control Presence": traffic_ctrl,
                    "Vehicle Type Involved": vehicle_type, "Driver Gender": gender,
                    "Driver License Status": license_status, "Alcohol Involvement": alcohol,
                    "Accident Location Details": location_det,
                    "Number of Vehicles Involved": num_vehicles,
                    "Number of Casualties": num_casualties,
                    "Number of Fatalities": num_fatalities,
                    "Speed Limit (km/h)": speed_limit, "Driver Age": driver_age,
                    "Hour": hour, "Month": month, "Day of Week": day_of_week,
                }
                with st.spinner("Predicting..."):
                    try:
                        result = _req.post(f"{API_BASE}/predict", json=payload, timeout=10).json()
                    except Exception as e:
                        st.error(f"Request failed: {e}"); st.stop()

                if "error" in result:
                    st.error(f"Backend error: {result['error']}")
                else:
                    label      = result["prediction"]
                    confidence = result["confidence"]
                    probs      = result["probabilities"]
                    css_cls    = f"result-{label.lower()}"
                    emoji      = {"Minor": "🟢", "Serious": "🟠", "Fatal": "🔴"}.get(label, "")
                    st.markdown(f"""
                    <div class="result-box {css_cls}">
                        <div class="metric-label">Predicted Severity</div>
                        <div class="metric-value">{emoji} {label}</div>
                        <div style="margin-top:.4rem;color:#57606a;">
                            Model confidence: <strong>{confidence}%</strong>
                        </div>
                    </div>""", unsafe_allow_html=True)
                    st.markdown("#### Probability Breakdown")
                    for cls, pct in sorted(probs.items(), key=lambda x: -x[1]):
                        st.markdown(f"**{cls}**  —  `{pct}%`")
                        st.progress(int(pct))
                    st.markdown("---")
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Vehicles",   num_vehicles)
                    m2.metric("Casualties", num_casualties)
                    m3.metric("Fatalities", num_fatalities)
            else:
                st.info("Fill the form and click **Predict Severity** to see results.")
                df_p = load_data()
                if not df_p.empty:
                    st.markdown("#### Dataset Quick Stats")
                    st.bar_chart(df_p["Accident Severity"].value_counts())

    # ── TAB 2 : FEATURE IMPORTANCE ────────────────────────────────────────────
    with tab2:
        st.subheader("Top Feature Importances (Random Forest)")
        st.caption("Higher importance = greater influence on the prediction.")
        fi = fetch_importance()
        if fi:
            fi_df = pd.DataFrame(fi).sort_values("importance")
            fi_df["feature"] = fi_df["feature"].str.replace("_enc", " (cat)").str.replace("_", " ")
            fig, ax = plt.subplots(figsize=(8, 5))
            bars = ax.barh(fi_df["feature"], fi_df["importance"], color="#3b82d4")
            ax.set_xlabel("Importance Score")
            ax.set_title("Random Forest Feature Importances")
            ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=8)
            ax.tick_params(axis="y", labelsize=9)
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.warning("Could not fetch feature importance data.")

    # ── TAB 3 : DATASET EXPLORER ──────────────────────────────────────────────
    with tab3:
        st.subheader("Raw Dataset Explorer")
        df = load_data()
        if df.empty:
            st.warning("Dataset not found.")
        else:
            f1, f2, f3 = st.columns(3)
            with f1: sel_state   = st.selectbox("Filter by State",    ["All"] + sorted(df["State Name"].unique().tolist()))
            with f2: sel_sev     = st.selectbox("Filter by Severity", ["All"] + sorted(df["Accident Severity"].unique().tolist()))
            with f3: sel_weather = st.selectbox("Filter by Weather",  ["All"] + sorted(df["Weather Conditions"].unique().tolist()))
            filtered = df.copy()
            if sel_state   != "All": filtered = filtered[filtered["State Name"]         == sel_state]
            if sel_sev     != "All": filtered = filtered[filtered["Accident Severity"]  == sel_sev]
            if sel_weather != "All": filtered = filtered[filtered["Weather Conditions"] == sel_weather]
            st.markdown(f"Showing **{len(filtered):,}** of **{len(df):,}** records")
            st.dataframe(filtered, use_container_width=True, height=400)
            st.markdown("---")
            c1, c2 = st.columns(2)
            with c1: st.markdown("**By Severity**");  st.bar_chart(filtered["Accident Severity"].value_counts())
            with c2: st.markdown("**By Weather**");   st.bar_chart(filtered["Weather Conditions"].value_counts())
            c3, c4 = st.columns(2)
            with c3: st.markdown("**By Road Type**"); st.bar_chart(filtered["Road Type"].value_counts())
            with c4:
                st.markdown("**By Month**")
                mo = ["January","February","March","April","May","June","July","August","September","October","November","December"]
                st.bar_chart(filtered["Month"].value_counts().reindex(mo).dropna())

    # ── TAB 4 : ABOUT ─────────────────────────────────────────────────────────
    with tab4:
        st.subheader("About This Project")
        st.markdown("""
        ## India Road Accident Severity Predictor
        Predicts whether a road accident results in **Minor**, **Serious**, or **Fatal** outcome.

        | Layer    | Technology                   |
        |----------|------------------------------|
        | ML Model | Random Forest (scikit-learn) |
        | Backend  | Flask + Flask-CORS  :5000    |
        | Frontend | Streamlit           :8501    |
        | Data     | accident_prediction_india.csv (3,000 records) |

        ### Severity Classes
        - 🟢 **Minor**   — Low impact
        - 🟠 **Serious** — Significant injuries
        - 🔴 **Fatal**   — One or more deaths

        ### How to Run (Single Command)
        ```
        python ALL_CODE.py
        ```
        Then open:  http://localhost:8501
        """)

    st.markdown("---")
    st.caption("India Accident Severity Predictor | Flask + Streamlit + scikit-learn")


# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================
def _start_flask_thread():
    """Start Flask backend in a background thread."""
    t = threading.Thread(target=run_flask_server, daemon=True)
    t.start()
    return t


def _start_streamlit_subprocess():
    """Launch Streamlit as a subprocess pointing to this same file."""
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["STREAMLIT_LAUNCHED_FROM_ALL_CODE"] = "1"
    proc = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run",
         os.path.abspath(__file__),
         "--server.port", "8501",
         "--server.headless", "false",
         "--browser.gatherUsageStats", "false"],
        env=env,
    )
    return proc


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else ""

    # ── python ALL_CODE.py train ───────────────────────────────────────────────
    if arg == "train":
        run_training()

    # ── python ALL_CODE.py serve ───────────────────────────────────────────────
    elif arg == "serve":
        run_flask_server()

    # ── python ALL_CODE.py  (no args) — start BOTH ────────────────────────────
    else:
        print("")
        print("=" * 55)
        print("  India Road Accident Severity Predictor")
        print("=" * 55)
        print("  Starting Flask backend  on http://127.0.0.1:5000 ...")
        print("  Starting Streamlit UI   on http://localhost:8501  ...")
        print("=" * 55)
        print("  Open your browser at:  http://localhost:8501")
        print("  Press Ctrl+C to stop both servers.")
        print("=" * 55)
        print("")

        # Start Flask in background thread
        _start_flask_thread()

        # Start Streamlit in foreground (blocks until Ctrl+C)
        proc = _start_streamlit_subprocess()
        try:
            proc.wait()
        except KeyboardInterrupt:
            proc.terminate()
            print("\nServers stopped.")

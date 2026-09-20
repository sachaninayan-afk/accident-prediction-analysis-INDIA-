# -*- coding: utf-8 -*-
# ==============================================================================
#  ALL_CODE.py
#  India Road Accident Severity Predictor
#  Contains the complete, unmodified source code of all three modules:
#
#  SECTION 1 (line  17) : model/train_model.py
#  SECTION 2 (line 163) : backend/app.py
#  SECTION 3 (line 331) : frontend/streamlit_app.py
#
#  Run instructions:
#    python ALL_CODE.py train    -> trains + saves the model
#    python ALL_CODE.py serve    -> starts the Flask backend on :5000
#    streamlit run ALL_CODE.py   -> launches the Streamlit frontend
# ==============================================================================


# ##############################################################################
# SECTION 1 — model/train_model.py
# Preprocesses the accident dataset, trains a Random Forest classifier
# to predict Accident Severity (Minor / Serious / Fatal), and saves
# the trained pipeline as model/accident_model.pkl.
# ##############################################################################

import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from flask import Flask, request, jsonify
from flask_cors import CORS

# ── Shared paths (relative to this file's location) ───────────────────────────
_HERE      = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(_HERE, "data", "accident_prediction_india.csv")
MODEL_PATH = os.path.join(_HERE, "model", "accident_model.pkl")
ENC_PATH   = os.path.join(_HERE, "model", "encoders.pkl")

# ── Shared lookup maps ─────────────────────────────────────────────────────────
MONTH_MAP = {
    "January": 1, "February": 2, "March": 3, "April": 4,
    "May": 5, "June": 6, "July": 7, "August": 8,
    "September": 9, "October": 10, "November": 11, "December": 12,
}
DAY_MAP = {
    "Monday": 1, "Tuesday": 2, "Wednesday": 3, "Thursday": 4,
    "Friday": 5, "Saturday": 6, "Sunday": 7,
}


def run_training():
    """
    train_model.py — full training pipeline.
    Loads the CSV, engineers features, trains RandomForest,
    evaluates on a test split, and saves model + encoders.
    """

    # ── Load data ──────────────────────────────────────────────────────────────
    df = pd.read_csv(DATA_PATH)
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

    # ── Feature engineering ────────────────────────────────────────────────────
    # Parse hour from "Time of Day" (format H:MM or HH:MM)
    df["Hour"] = df["Time of Day"].apply(
        lambda t: int(str(t).split(":")[0]) if pd.notna(t) else 12
    )

    # Map Month to numeric
    month_map = {
        "January": 1, "February": 2, "March": 3, "April": 4,
        "May": 5, "June": 6, "July": 7, "August": 8,
        "September": 9, "October": 10, "November": 11, "December": 12,
    }
    df["Month_Num"] = df["Month"].map(month_map).fillna(6).astype(int)

    # Map Day of Week to numeric
    day_map = {
        "Monday": 1, "Tuesday": 2, "Wednesday": 3, "Thursday": 4,
        "Friday": 5, "Saturday": 6, "Sunday": 7,
    }
    df["Day_Num"] = df["Day of Week"].map(day_map).fillna(1).astype(int)

    # ── Select features ────────────────────────────────────────────────────────
    CATEGORICAL = [
        "Weather Conditions",
        "Road Type",
        "Road Condition",
        "Lighting Conditions",
        "Traffic Control Presence",
        "Vehicle Type Involved",
        "Driver Gender",
        "Driver License Status",
        "Alcohol Involvement",
        "Accident Location Details",
    ]

    NUMERICAL = [
        "Number of Vehicles Involved",
        "Number of Casualties",
        "Number of Fatalities",
        "Speed Limit (km/h)",
        "Driver Age",
        "Hour",
        "Month_Num",
        "Day_Num",
    ]

    TARGET = "Accident Severity"

    # Drop rows with missing target
    df = df.dropna(subset=[TARGET])

    # ── Encode categoricals ────────────────────────────────────────────────────
    encoders = {}
    for col in CATEGORICAL:
        le = LabelEncoder()
        df[col + "_enc"] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    # Encode target
    target_le = LabelEncoder()
    df["Target"] = target_le.fit_transform(df[TARGET])
    encoders["__target__"] = target_le

    print("Target classes:", list(target_le.classes_))

    # ── Build feature matrix ───────────────────────────────────────────────────
    enc_cols = [c + "_enc" for c in CATEGORICAL]
    feature_cols = NUMERICAL + enc_cols

    X = df[feature_cols].fillna(0).astype(float)
    y = df["Target"]

    print(f"Features: {len(feature_cols)}  |  Samples: {len(X)}")

    # ── Train / test split ─────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ── Train model ────────────────────────────────────────────────────────────
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # ── Evaluate ───────────────────────────────────────────────────────────────
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=target_le.classes_))

    # ── Save artefacts ─────────────────────────────────────────────────────────
    meta = {
        "feature_cols": feature_cols,
        "categorical":  CATEGORICAL,
        "numerical":    NUMERICAL,
    }

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    with open(ENC_PATH, "wb") as f:
        pickle.dump({"encoders": encoders, "meta": meta}, f)

    print(f"\nModel saved  -> {MODEL_PATH}")
    print(f"Encoders saved -> {ENC_PATH}")


# ##############################################################################
# SECTION 2 — backend/app.py
# Flask REST API backend.
# Endpoints:
#   GET  /health             — health check
#   POST /predict            — predict accident severity from JSON payload
#   GET  /options            — return dropdown option lists for the frontend
#   GET  /feature_importance — top-N feature importances from the model
# ##############################################################################

# ── Option lists ──────────────────────────────────────────────────────────────
OPTIONS = {
    "Weather Conditions":       ["Clear", "Rainy", "Foggy", "Hazy", "Stormy"],
    "Road Type":                ["National Highway", "State Highway", "Urban Road", "Village Road"],
    "Road Condition":           ["Dry", "Wet", "Damaged", "Under Construction"],
    "Lighting Conditions":      ["Daylight", "Dark", "Dawn", "Dusk"],
    "Traffic Control Presence": ["Signs", "Signals", "Police Checkpost", "None"],
    "Vehicle Type Involved":    ["Car", "Truck", "Bus", "Two-Wheeler", "Cycle", "Auto-Rickshaw", "Pedestrian"],
    "Driver Gender":            ["Male", "Female"],
    "Driver License Status":    ["Valid", "Expired", "None"],
    "Alcohol Involvement":      ["Yes", "No"],
    "Accident Location Details":["Straight Road", "Curve", "Intersection", "Bridge"],
}


def _load_model_bundle():
    """Load the trained model and encoders from disk."""
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(ENC_PATH, "rb") as f:
        bundle   = pickle.load(f)
        encoders = bundle["encoders"]
        meta     = bundle["meta"]
    return model, encoders, meta


def build_feature_vector(data: dict, encoders: dict, feature_cols: list,
                          categorical: list) -> pd.DataFrame:
    """Convert a raw input dict into the model's feature vector (DataFrame)."""
    row = {}

    # Numerical
    row["Number of Vehicles Involved"] = float(data.get("Number of Vehicles Involved", 2))
    row["Number of Casualties"]        = float(data.get("Number of Casualties", 0))
    row["Number of Fatalities"]        = float(data.get("Number of Fatalities", 0))
    row["Speed Limit (km/h)"]          = float(data.get("Speed Limit (km/h)", 60))
    row["Driver Age"]                  = float(data.get("Driver Age", 30))

    # Hour from "Time of Day" string or direct "Hour"
    if "Hour" in data:
        row["Hour"] = float(data["Hour"])
    elif "Time of Day" in data:
        t = str(data["Time of Day"])
        row["Hour"] = float(t.split(":")[0])
    else:
        row["Hour"] = 12.0

    row["Month_Num"] = float(MONTH_MAP.get(data.get("Month", "June"), 6))
    row["Day_Num"]   = float(DAY_MAP.get(data.get("Day of Week", "Monday"), 1))

    # Categorical — label-encode with the saved encoders
    for col in categorical:
        le  = encoders[col]
        val = str(data.get(col, le.classes_[0]))
        if val not in le.classes_:
            val = le.classes_[0]
        row[col + "_enc"] = float(le.transform([val])[0])

    # Assemble in the exact order the model was trained on (keep feature names for sklearn)
    vector = [row[fc] for fc in feature_cols]
    return pd.DataFrame([vector], columns=feature_cols)


def create_flask_app() -> Flask:
    """
    app.py — creates and returns the configured Flask application.
    Loads model + encoders, registers all routes, enables CORS.
    """
    model, encoders, meta = _load_model_bundle()
    target_le    = encoders["__target__"]
    feature_cols = meta["feature_cols"]
    categorical  = meta["categorical"]

    app = Flask(__name__)
    CORS(app)

    # ── GET /health ────────────────────────────────────────────────────────────
    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({
            "status":  "ok",
            "model":   "RandomForestClassifier",
            "classes": list(target_le.classes_),
        })

    # ── GET /options ───────────────────────────────────────────────────────────
    @app.route("/options", methods=["GET"])
    def options():
        return jsonify({
            "categorical_options": OPTIONS,
            "months": list(MONTH_MAP.keys()),
            "days":   list(DAY_MAP.keys()),
        })

    # ── POST /predict ──────────────────────────────────────────────────────────
    @app.route("/predict", methods=["POST"])
    def predict():
        payload = request.get_json(force=True)
        if not payload:
            return jsonify({"error": "Empty or invalid JSON body"}), 400

        try:
            X       = build_feature_vector(payload, encoders, feature_cols, categorical)
            pred_id = model.predict(X)[0]
            proba   = model.predict_proba(X)[0]
            label   = target_le.inverse_transform([pred_id])[0]

            probabilities = {
                cls: round(float(p) * 100, 2)
                for cls, p in zip(target_le.classes_, proba)
            }

            # Severity colour hint
            colour_map = {"Minor": "green", "Serious": "orange", "Fatal": "red"}

            return jsonify({
                "prediction":    label,
                "confidence":    round(float(max(proba)) * 100, 2),
                "probabilities": probabilities,
                "colour":        colour_map.get(label, "grey"),
            })
        except Exception as exc:
            return jsonify({"error": str(exc)}), 500

    # ── GET /feature_importance ────────────────────────────────────────────────
    @app.route("/feature_importance", methods=["GET"])
    def feature_importance():
        n           = int(request.args.get("top", 10))
        importances = model.feature_importances_
        indices     = np.argsort(importances)[::-1][:n]
        result = [
            {"feature": feature_cols[i], "importance": round(float(importances[i]), 4)}
            for i in indices
        ]
        return jsonify(result)

    return app


def run_flask_server():
    """Entry point: start the Flask backend on port 5000."""
    app = create_flask_app()
    print("Flask backend running on http://127.0.0.1:5000")
    app.run(debug=False, port=5000, use_reloader=False)


# ##############################################################################
# SECTION 3 — frontend/streamlit_app.py
# Streamlit interactive frontend.
# Tabs: Predict | Feature Importance | Dataset Explorer | About
#
# To launch:  streamlit run ALL_CODE.py
# ##############################################################################

# Streamlit only runs its top-level code when executed via `streamlit run`.
# The guard below ensures Flask/training code does NOT run in that context,
# and the Streamlit UI does NOT render when running plain `python ALL_CODE.py`.

def _is_streamlit() -> bool:
    """Return True when running inside a Streamlit session."""
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except Exception:
        return False


if _is_streamlit():
    import streamlit as st

    # ── Config ──────────────────────────────────────────────────────────────────
    API_BASE       = "http://127.0.0.1:5000"
    ST_DATA_PATH   = os.path.join(_HERE, "data", "accident_prediction_india.csv")

    st.set_page_config(
        page_title="India Accident Severity Predictor",
        page_icon="🚦",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ── Custom CSS ───────────────────────────────────────────────────────────────
    st.markdown("""
    <style>
        .main-header {
            font-size: 2.2rem;
            font-weight: 700;
            color: #1f2328;
            margin-bottom: 0.2rem;
        }
        .sub-header {
            font-size: 1rem;
            color: #57606a;
            margin-bottom: 1.5rem;
        }
        .result-box {
            padding: 1.4rem 1.8rem;
            border-radius: 10px;
            border-left: 6px solid;
            margin-top: 1rem;
        }
        .result-minor   { border-color: #22c55e; background: #f0fdf4; }
        .result-serious { border-color: #f97316; background: #fff7ed; }
        .result-fatal   { border-color: #ef4444; background: #fef2f2; }
        .metric-label   { font-size: 0.85rem; color: #57606a; font-weight: 600; }
        .metric-value   { font-size: 1.8rem; font-weight: 700; }
        .stButton > button {
            background: #3b82d4;
            color: white;
            border: none;
            border-radius: 6px;
            padding: 0.5rem 2rem;
            font-size: 1rem;
            font-weight: 600;
        }
        .stButton > button:hover { background: #2563eb; }
    </style>
    """, unsafe_allow_html=True)

    # ── Header ───────────────────────────────────────────────────────────────────
    st.markdown('<div class="main-header">🚦 India Road Accident Severity Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Enter accident scenario details to predict severity — Minor, Serious, or Fatal</div>', unsafe_allow_html=True)
    st.divider()

    # ── Fetch options from backend ────────────────────────────────────────────────
    import requests as _requests_module
    requests = _requests_module

    @st.cache_data(show_spinner=False)
    def fetch_options():
        try:
            r = requests.get(f"{API_BASE}/options", timeout=5)
            return r.json()
        except Exception:
            return None

    @st.cache_data(show_spinner=False)
    def fetch_feature_importance():
        try:
            r = requests.get(f"{API_BASE}/feature_importance?top=12", timeout=5)
            return r.json()
        except Exception:
            return []

    @st.cache_data(show_spinner=False)
    def load_dataset():
        try:
            return pd.read_csv(ST_DATA_PATH)
        except Exception:
            return pd.DataFrame()

    opts = fetch_options()
    if opts is None:
        st.error("Cannot connect to Flask backend at http://127.0.0.1:5000 — please start it first (`python ALL_CODE.py serve`).")
        st.stop()

    cat_opts = opts["categorical_options"]
    months   = opts["months"]
    days     = opts["days"]

    # ── Tabs ──────────────────────────────────────────────────────────────────────
    tab_predict, tab_importance, tab_data, tab_about = st.tabs([
        "🔮 Predict", "📊 Feature Importance", "📂 Dataset Explorer", "ℹ️ About"
    ])

    # ═══════════════════════════════════════════════════════════════════════════════
    # TAB 1 — PREDICT
    # ═══════════════════════════════════════════════════════════════════════════════
    with tab_predict:
        col_form, col_result = st.columns([1.1, 0.9], gap="large")

        with col_form:
            st.subheader("Accident Scenario Details")

            with st.form("prediction_form"):
                r1c1, r1c2 = st.columns(2)
                with r1c1:
                    weather       = st.selectbox("Weather Conditions",       cat_opts["Weather Conditions"])
                    road_type     = st.selectbox("Road Type",                cat_opts["Road Type"])
                    road_cond     = st.selectbox("Road Condition",           cat_opts["Road Condition"])
                    lighting      = st.selectbox("Lighting Conditions",      cat_opts["Lighting Conditions"])
                    traffic_ctrl  = st.selectbox("Traffic Control Presence", cat_opts["Traffic Control Presence"])

                with r1c2:
                    vehicle_type   = st.selectbox("Vehicle Type Involved",    cat_opts["Vehicle Type Involved"])
                    gender         = st.selectbox("Driver Gender",            cat_opts["Driver Gender"])
                    license_status = st.selectbox("Driver License Status",    cat_opts["Driver License Status"])
                    alcohol        = st.selectbox("Alcohol Involvement",      cat_opts["Alcohol Involvement"])
                    location_det   = st.selectbox("Accident Location",        cat_opts["Accident Location Details"])

                st.markdown("---")
                r2c1, r2c2, r2c3 = st.columns(3)
                with r2c1:
                    num_vehicles   = st.slider("Number of Vehicles",   1, 10, 2)
                    num_casualties = st.slider("Number of Casualties", 0, 20, 2)
                    num_fatalities = st.slider("Number of Fatalities", 0, 10, 0)
                with r2c2:
                    speed_limit    = st.slider("Speed Limit (km/h)",   20, 140, 60)
                    driver_age     = st.slider("Driver Age",           16, 80,  30)
                    hour           = st.slider("Hour of Day",          0, 23,   12)
                with r2c3:
                    month          = st.selectbox("Month", months)
                    day_of_week    = st.selectbox("Day",   days)

                submitted = st.form_submit_button("🔮  Predict Severity", use_container_width=True)

        with col_result:
            st.subheader("Prediction Result")

            if submitted:
                payload = {
                    "Weather Conditions":          weather,
                    "Road Type":                   road_type,
                    "Road Condition":              road_cond,
                    "Lighting Conditions":         lighting,
                    "Traffic Control Presence":    traffic_ctrl,
                    "Vehicle Type Involved":       vehicle_type,
                    "Driver Gender":               gender,
                    "Driver License Status":       license_status,
                    "Alcohol Involvement":         alcohol,
                    "Accident Location Details":   location_det,
                    "Number of Vehicles Involved": num_vehicles,
                    "Number of Casualties":        num_casualties,
                    "Number of Fatalities":        num_fatalities,
                    "Speed Limit (km/h)":          speed_limit,
                    "Driver Age":                  driver_age,
                    "Hour":                        hour,
                    "Month":                       month,
                    "Day of Week":                 day_of_week,
                }

                with st.spinner("Predicting..."):
                    try:
                        resp   = requests.post(f"{API_BASE}/predict", json=payload, timeout=10)
                        result = resp.json()
                    except Exception as e:
                        st.error(f"Request failed: {e}")
                        st.stop()

                if "error" in result:
                    st.error(f"Backend error: {result['error']}")
                else:
                    label      = result["prediction"]
                    confidence = result["confidence"]
                    probs      = result["probabilities"]
                    css_cls    = f"result-{label.lower()}"

                    # Main result box
                    colour_emoji = {"Minor": "🟢", "Serious": "🟠", "Fatal": "🔴"}
                    st.markdown(f"""
                    <div class="result-box {css_cls}">
                        <div class="metric-label">Predicted Severity</div>
                        <div class="metric-value">{colour_emoji.get(label, "")} {label}</div>
                        <div style="margin-top:0.4rem;color:#57606a;">
                            Model confidence: <strong>{confidence}%</strong>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown("#### Probability Breakdown")
                    for cls, pct in sorted(probs.items(), key=lambda x: -x[1]):
                        st.markdown(f"**{cls}**  —  `{pct}%`")
                        st.progress(int(pct))

                    # Summary metrics
                    st.markdown("---")
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Vehicles",   num_vehicles)
                    m2.metric("Casualties", num_casualties)
                    m3.metric("Fatalities", num_fatalities)
            else:
                st.info("Fill the form on the left and click **Predict Severity** to see results.")

                # Show a quick stats preview
                st.markdown("#### Dataset Quick Stats")
                df_preview = load_dataset()
                if not df_preview.empty:
                    sev_counts = df_preview["Accident Severity"].value_counts()
                    st.bar_chart(sev_counts)

    # ═══════════════════════════════════════════════════════════════════════════════
    # TAB 2 — FEATURE IMPORTANCE
    # ═══════════════════════════════════════════════════════════════════════════════
    with tab_importance:
        st.subheader("Top Feature Importances (Random Forest)")
        st.caption("Higher importance = greater influence on the prediction.")

        fi_data = fetch_feature_importance()
        if fi_data:
            fi_df = pd.DataFrame(fi_data).sort_values("importance")
            fi_df["feature"] = fi_df["feature"].str.replace("_enc", " (cat)").str.replace("_", " ")

            fig, ax = plt.subplots(figsize=(8, 5))
            bars = ax.barh(fi_df["feature"], fi_df["importance"], color="#3b82d4")
            ax.set_xlabel("Importance Score")
            ax.set_title("Random Forest Feature Importances")
            ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=8)
            ax.tick_params(axis='y', labelsize=9)
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.warning("Could not fetch feature importance data from backend.")

    # ═══════════════════════════════════════════════════════════════════════════════
    # TAB 3 — DATASET EXPLORER
    # ═══════════════════════════════════════════════════════════════════════════════
    with tab_data:
        st.subheader("Raw Dataset Explorer")
        df = load_dataset()

        if df.empty:
            st.warning("Dataset not found.")
        else:
            # Filters
            fc1, fc2, fc3 = st.columns(3)
            with fc1:
                states    = ["All"] + sorted(df["State Name"].unique().tolist())
                sel_state = st.selectbox("Filter by State", states)
            with fc2:
                sevs    = ["All"] + sorted(df["Accident Severity"].unique().tolist())
                sel_sev = st.selectbox("Filter by Severity", sevs)
            with fc3:
                weathers    = ["All"] + sorted(df["Weather Conditions"].unique().tolist())
                sel_weather = st.selectbox("Filter by Weather", weathers)

            filtered = df.copy()
            if sel_state   != "All": filtered = filtered[filtered["State Name"]         == sel_state]
            if sel_sev     != "All": filtered = filtered[filtered["Accident Severity"]  == sel_sev]
            if sel_weather != "All": filtered = filtered[filtered["Weather Conditions"] == sel_weather]

            st.markdown(f"Showing **{len(filtered):,}** of **{len(df):,}** records")
            st.dataframe(filtered, use_container_width=True, height=400)

            # Charts
            st.markdown("---")
            cc1, cc2 = st.columns(2)
            with cc1:
                st.markdown("**Accidents by Severity**")
                st.bar_chart(filtered["Accident Severity"].value_counts())
            with cc2:
                st.markdown("**Accidents by Weather**")
                st.bar_chart(filtered["Weather Conditions"].value_counts())

            cc3, cc4 = st.columns(2)
            with cc3:
                st.markdown("**Accidents by Road Type**")
                st.bar_chart(filtered["Road Type"].value_counts())
            with cc4:
                st.markdown("**Accidents by Month**")
                month_order  = ["January","February","March","April","May","June",
                                "July","August","September","October","November","December"]
                month_counts = filtered["Month"].value_counts().reindex(month_order).dropna()
                st.bar_chart(month_counts)

    # ═══════════════════════════════════════════════════════════════════════════════
    # TAB 4 — ABOUT
    # ═══════════════════════════════════════════════════════════════════════════════
    with tab_about:
        st.subheader("About This Project")
        st.markdown("""
        ## India Road Accident Severity Predictor

        This application predicts whether a road accident will result in a **Minor**, **Serious**, or **Fatal** outcome
        based on environmental, road, and driver characteristics.

        ### Architecture
        | Layer       | Technology              |
        |-------------|-------------------------|
        | ML Model    | Random Forest (scikit-learn) |
        | Backend API | Flask + Flask-CORS      |
        | Frontend    | Streamlit               |
        | Data        | accident_prediction_india.csv (3,000 records) |

        ### Dataset Features Used
        | Category       | Features |
        |----------------|----------|
        | Environment    | Weather Conditions, Lighting Conditions |
        | Road           | Road Type, Road Condition, Speed Limit |
        | Accident       | Number of Vehicles, Casualties, Fatalities, Location |
        | Driver         | Age, Gender, License Status, Alcohol Involvement |
        | Time           | Hour of Day, Day of Week, Month |

        ### Severity Classes
        - 🟢 **Minor** — Low impact, no fatalities
        - 🟠 **Serious** — Significant injuries
        - 🔴 **Fatal** — One or more deaths

        ### How to Run
        ```bash
        # 1. Install dependencies
        pip install -r requirements.txt

        # 2. Train the model
        python ALL_CODE.py train

        # 3. Start Flask backend  (Terminal 1)
        python ALL_CODE.py serve

        # 4. Start Streamlit frontend  (Terminal 2)
        streamlit run ALL_CODE.py
        ```
        """)

    # ── Footer ────────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.caption("India Accident Severity Predictor | Built with Flask + Streamlit + scikit-learn")


# ##############################################################################
# MAIN ENTRY POINT (python ALL_CODE.py)
# ##############################################################################
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        run_training()
    elif len(sys.argv) > 1 and sys.argv[1] == "serve":
        run_flask_server()
    else:
        print("=" * 60)
        print("  India Road Accident Severity Predictor — ALL_CODE.py")
        print("=" * 60)
        print()
        print("Usage:")
        print("  python ALL_CODE.py train    # Train + save the model")
        print("  python ALL_CODE.py serve    # Start Flask backend (:5000)")
        print("  streamlit run ALL_CODE.py   # Launch Streamlit frontend")
        print()
        print("All three sections (train_model / app / streamlit_app)")
        print("are contained verbatim in this single file.")

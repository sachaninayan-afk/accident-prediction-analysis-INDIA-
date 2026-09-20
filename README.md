# India Road Accident Severity Predictor

A full-stack machine learning application that predicts road accident severity
(**Minor / Serious / Fatal**) from environmental, road, and driver features using
India accident data.

ML          scikit-learn (Random Forest)
Backend     Flask, Flask-CORS
Frontend    Streamlit, Matplotlib
Dataset     https://www.kaggle.com/datasets/khushikyad001/india-road-accident-dataset-predictive-analysis

---

## Project Structure

```
accident_prediction_project/
│
├── ALL_CODE.py                         # Single file — all source code (train + backend + frontend)
│
├── data/
│   └── accident_prediction_india.csv   # Source dataset (3,000 records)
│
├── model/
│   ├── accident_model.pkl              # Trained Random Forest model (generated)
│   └── encoders.pkl                    # Label encoders + feature metadata (generated)
│
├── requirements.txt                    # Python dependencies
└── README.md
```

> All Python source code — model training, Flask backend, and Streamlit frontend —
> is contained in the single file **`ALL_CODE.py`**.

---

## Quickstart

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Model  *(generates model/accident_model.pkl)*
```bash
python ALL_CODE.py train
```

### 3. Start the Flask Backend  *(Terminal 1)*
```bash
python ALL_CODE.py serve
```
Backend runs at **http://127.0.0.1:5000**

### 4. Start the Streamlit Frontend  *(Terminal 2)*
```bash
streamlit run ALL_CODE.py
```
Frontend opens at **http://localhost:8501**

---

## ALL_CODE.py — Section Map

| Section   | Content                          | Trigger                          |
|-----------|----------------------------------|----------------------------------|
| Section 1 | Model training pipeline          | `python ALL_CODE.py train`       |
| Section 2 | Flask REST API (backend)         | `python ALL_CODE.py serve`       |
| Section 3 | Streamlit interactive UI         | `streamlit run ALL_CODE.py`      |

---

## API Endpoints

| Method | Endpoint              | Description                            |
|--------|-----------------------|----------------------------------------|
| GET    | `/health`             | Health check + model info              |
| GET    | `/options`            | Returns dropdown option lists          |
| POST   | `/predict`            | Predicts severity from JSON payload    |
| GET    | `/feature_importance` | Top-N feature importances              |

### Example `/predict` Request
```json
{
  "Weather Conditions": "Rainy",
  "Road Type": "National Highway",
  "Road Condition": "Wet",
  "Lighting Conditions": "Dark",
  "Traffic Control Presence": "Signs",
  "Vehicle Type Involved": "Truck",
  "Driver Gender": "Male",
  "Driver License Status": "Valid",
  "Alcohol Involvement": "Yes",
  "Accident Location Details": "Curve",
  "Number of Vehicles Involved": 3,
  "Number of Casualties": 5,
  "Number of Fatalities": 2,
  "Speed Limit (km/h)": 80,
  "Driver Age": 42,
  "Hour": 23,
  "Month": "December",
  "Day of Week": "Friday"
}
```

### Example Response
```json
{
  "prediction": "Fatal",
  "confidence": 45.5,
  "probabilities": {
    "Fatal": 45.5,
    "Serious": 32.0,
    "Minor": 22.5
  },
  "colour": "red"
}
```

---

## Machine Learning Model

| Property         | Value                       |
|------------------|-----------------------------|
| Algorithm        | Random Forest Classifier    |
| Trees            | 200                         |
| Max Depth        | 15                          |
| Train/Test Split | 80% / 20%                   |
| Target Classes   | Minor, Serious, Fatal       |
| Feature Count    | 18                          |
| Dataset Size     | 3,000 records               |

---

## Frontend Features

- **Predict tab** — Interactive form with dropdowns and sliders, instant prediction result with colour-coded severity badge and probability breakdown
- **Feature Importance tab** — Horizontal bar chart of top model features
- **Dataset Explorer tab** — Filterable, sortable data table + 4 insight charts
- **About tab** — Architecture summary and run instructions

---

## Tech Stack

| Layer    | Technology                   |
|----------|------------------------------|
| ML       | scikit-learn (Random Forest) |
| Backend  | Flask, Flask-CORS            |
| Frontend | Streamlit, Matplotlib        |
| Data     | Pandas, NumPy                |

# 🏙️ Bangalore House Price Prediction

![Python](https://img.shields.io/badge/Python-3.13-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3-orange.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-Optimized-green.svg)
![Status](https://img.shields.io/badge/Status-Datathon_Ready-brightgreen.svg)

> **Official Submission for DataVerse Datathon 2025**  
> **Track 2:** Bengaluru House Prices  

This repository contains an end-to-end machine-learning dashboard for predicting Bengaluru property prices. It combines engineered property features, metro proximity analysis, and a stacked regression model in an interactive Streamlit application.

---

## 📐 System Architecture & Pipeline

We approached this problem using an iterative, two-phase pipeline architecture, culminating in a dynamic technical pivot.

### 1. Phase 1: The Interpretable Baseline
We established our ground truth using a **Linear Regression** model.
* **Feature Engineering:** Extracted `total_rooms` and `price_per_sqft`.
* **Noise Reduction:** Implemented Interquartile Range (IQR) filtering to remove extreme luxury property spikes that skewed the linear plane.
* *Result:* Established a strong, interpretable mathematical baseline.

### 2. Phase 2: The Advanced Stacking Ensemble
Realizing the dataset contained complex, non-linear relationships (especially regarding location premiums), we upgraded our architecture to an **Advanced Stacking Regressor**.
* **Base Models:** `XGBRegressor`, `HistGradientBoostingRegressor`, and `RandomForestRegressor`.
* **Meta-Model:** `Ridge Regression` to synthesize the final predictions.
* **Mathematical Optimization:** Target variable (`price`) was transformed using `np.log1p()` to strictly optimize the competition's evaluation metric (RMSLE) natively during training.

### 3. The Sudden Death Pivot: Infrastructure Proximity
At 3:00 PM, we integrated live spatial coordinates to solve the technical pivot.
* We mapped the 20 regional text strings to raw Latitude/Longitude coordinates.
* We applied the geometric **Haversine Formula** to calculate the exact radial distance (in km) from every property to the central **MG Road Metro Station Hub**.
* **Pivot Insight:** We mathematically proved a negative correlation (-0.0817), confirming that as distance to the metro hub *increases*, property prices *decrease*. This new dynamic feature was successfully integrated into the Stacking Ensemble.

---

## 📊 Key Artifacts & Visual Proof

Our pipeline automatically generates high-resolution audit artifacts to ensure total transparency of our ML lifecycle:
1. `distribution_plot_audit.png`: Proves the statistical normalization of the target variable via log transformation.
2. `pivot_metro_analysis.png`: Scatterplot proving the negative correlation between Metro Distance and Price.
3. `stacked_feature_importance.png`: Visualizes the logic driving the Ensemble's decision trees.

---

## 🚀 Run the dashboard locally

**1. Clone the Repository:**
```bash
git clone https://github.com/hanamaraddi9620adi/BangaloreHousePricePrediction.git
cd BangaloreHousePricePrediction
```

**2. Install Dependencies:**
```bash
pip install -r requirements.txt
```

**3. Start Streamlit:**
```bash
streamlit run app.py
```

The dashboard loads the committed `stacked_model.pkl` and `house_prices_bangalore.csv` files. The model-training scripts in `src/` can be run separately when retraining is required.

## ☁️ Deploy on Streamlit Community Cloud

1. Open [share.streamlit.io](https://share.streamlit.io/) and sign in with GitHub.
2. Select the `hanamaraddi9620adi/BangaloreHousePricePrediction` repository and its `main` branch.
3. Set the main file path to `app.py`, then click **Deploy**.

The repository includes `requirements.txt`, `runtime.txt`, and `.streamlit/config.toml` so the hosted environment uses the same dependency and theme configuration.

## ⚡ Deploy on Vercel

The Vercel deployment uses the lightweight web dashboard in `index.html` and the
dependency-free Python API in `api/index.py`. It avoids the large scientific
Python bundle required by Streamlit and loads quickly on Vercel serverless
functions.

1. Import this repository at [vercel.com/new](https://vercel.com/new).
2. Keep the framework preset as **Other**.
3. Use the root directory and click **Deploy**.

The API is available at `/api/health` and `/api/predict`. The original
Streamlit dashboard can still be run locally with the dependencies in
`requirements-streamlit.txt`.

---

## 🏆 Final Model Metrics

* **Architecture:** Ridge-Stacked Meta-Ensemble (XGBoost + HistGradientBoosting + Random Forest)
* **Optimization Metric:** RMSLE via `np.log1p()`
* **True Baseline R²:** 60.87% *(Note: Achieved mathematically without committing Target Leakage on `price_per_sqft`)*.

---

*Built with engineering discipline by the Prime Team.*

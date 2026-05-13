# 🏙️ DataVerse: Bengaluru House Price Prediction Engine

![Python](https://img.shields.io/badge/Python-3.13-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3-orange.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-Optimized-green.svg)
![Status](https://img.shields.io/badge/Status-Datathon_Ready-brightgreen.svg)

> **Official Submission for DataVerse Datathon 2025**  
> **Track 2:** Bengaluru House Prices  

This repository contains our end-to-end Machine Learning pipeline designed to predict luxury and standard property prices in Bengaluru. We heavily prioritized engineering discipline, feature extraction, and mathematically robust model stacking to minimize **RMSLE (Root Mean Squared Logarithmic Error)**.

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

## 🚀 How to Run the Pipeline Locally

**1. Clone the Repository:**
```bash
git clone https://github.com/lochangowda10/DataVerse.git
cd DataVerse
```

**2. Install Dependencies:**
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
```

**3. Execute the Full Pipeline (with Pivot Integration):**
```bash
cd src
python model_training.py
```
*This script will automatically load the dataset, apply the Haversine pivot mapping, train the Stacking Ensemble, and output the final R² and RMSLE metrics.*

---

## 🏆 Final Model Metrics

* **Architecture:** Ridge-Stacked Meta-Ensemble (XGBoost + HistGradientBoosting + Random Forest)
* **Optimization Metric:** RMSLE via `np.log1p()`
* **True Baseline R²:** 60.87% *(Note: Achieved mathematically without committing Target Leakage on `price_per_sqft`)*.

---

*Built with engineering discipline by the Prime Team.*

# 🏎️ F1 Pit-Wall Telemetry & Strategy Suite

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![Plotly](https://img.shields.io/badge/Plotly-Interactive%20Visuals-239120.svg)](https://plotly.com/)
[![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-F7931E.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An enterprise-grade Formula 1 race engineering workstation and telemetry analytics platform combining **Real-World GPS Spline Geometry**, **Non-Linear State Estimation (EKF)**, **Supervised Machine Learning Strategy Optimization**, **Computer Vision Kinematic Tracking**, and **Unsupervised Time-Series Anomaly Detection**.

---

## 🌟 The 4 Core Race Engineering Pillars

### 1. 🧠 Predictive Race Strategy & Lap-Time Regression (Machine Learning)
- **Problem:** Race strategy requires estimating non-linear tyre thermal wear, fuel burn advantages, and pit-stop deltas under uncertain track conditions.
- **Implementation:** Random Forest and Gradient Boosting regressors trained on multi-stint compound telemetry (Soft C4, Medium C3, Hard C2).
- **Features:** Lap-by-lap pace prediction, compound cliff modeling, and a 1,500-run Monte Carlo stochastic simulator evaluating 1-Stop vs. 2-Stop win probabilities under Safety Car interventions.

### 2. 📐 Vehicle Dynamics & State Estimation (Extended Kalman Filter)
- **Problem:** Critical dynamic vehicle states such as **Sideslip Angle ($\beta$)** and tire slip angles cannot be directly measured by physical sensors.
- **Implementation:** Non-linear dynamic bicycle model fused with noisy IMU data (lateral acceleration $a_y$, yaw rate $r$) and GPS velocity via an **Extended Kalman Filter (EKF)**.
- **Output:** Latent sideslip angle tracking, friction circle (G-G diagram) mapping, and estimator covariance convergence.

### 3. 👁️ Motorsport Spatial Motion Analysis & Computer Vision Calibration
- **Problem:** Camera-to-world extrinsic calibration in motorsport video requires tracking visual apparent motion and braking kinematics.
- **Implementation:** 2D optical-flow vector field decomposition $(u, v)$, apex curvature peak identification, and deceleration profile extraction.
- **Output:** Corner apex detection, visual shear angles, and braking point thresholding.

### 4. ⚠️ Time-Series Anomaly Detection (Predictive Maintenance)
- **Problem:** Mechanical faults (gearbox bearing flaking, hydraulic pressure leaks, MGU-K thermal runaway) produce subtle anomalies before catastrophic DNF.
- **Implementation:** Unsupervised **Isolation Forest** with rolling residual scoring across multi-variate high-frequency telemetry.
- **Output:** Real-time component health index ($0-100\%$), risk score timeline, and early warning fault triggers before mechanical retirement.

---

## 🚀 Quick Start Guide

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/f1-pitwall-strategy-suite.git
cd f1-pitwall-strategy-suite
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch Streamlit Web Suite
```bash
streamlit run app.py
```
*Open your browser at `http://localhost:8501` to access all 6 interactive tabs.*

### 4. Launch Desktop Animation HUD (Optional)
```bash
python pitwall_desktop.py
```
*Hotkeys: `[TAB]` next track, `[SHIFT+TAB]` prev track, `[1-9]` direct jump, `[SPACE]` pause.*

---

## 💼 How to Present This on Your CV / Resume

```markdown
**F1 Race Engineering & Telemetry Analytics Suite** | *Python, Streamlit, Scikit-Learn, Plotly, FastF1, SciPy*
- Engineered an end-to-end Formula 1 race engineering workstation integrating machine learning strategy optimization, vehicle dynamics state estimation, and time-series anomaly detection.
- Developed an Extended Kalman Filter (EKF) sensor-fusion algorithm on a non-linear bicycle model to estimate latent vehicle sideslip angle (β) and yaw rate from noisy IMU and GPS telemetry.
- Built Random Forest regression models for tyre thermal degradation and a 1,500-iteration Monte Carlo race simulator to identify optimal pit stop windows under stochastic safety car conditions.
- Implemented unsupervised Isolation Forest models on multi-sensor CAN-bus data (vibration RMS, oil temperature, hydraulic pressure) for predictive maintenance and early mechanical fault detection.
```

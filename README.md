# 🏎️ F1 Pit-Wall Telemetry, ECU Firmware & Strategy Suite

[![C99 / MISRA-C:2012](https://img.shields.io/badge/C-C99%20%2F%20MISRA--C%3A2012-00599C.svg?logo=c)](https://en.wikipedia.org/wiki/MISRA_C)
[![CAN-Bus 2.0B / CAN-FD](https://img.shields.io/badge/Bus-CAN%202.0B%20%2F%20CAN--FD-orange.svg)](https://www.iso.org/standard/63648.html)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B.svg?logo=streamlit)](https://streamlit.io/)
[![Plotly](https://img.shields.io/badge/Plotly-Interactive%20Visuals-239120.svg?logo=plotly)](https://plotly.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An enterprise-grade Formula 1 engineering suite combining **On-Car Hard Real-Time Embedded C ECU Firmware (CAN 2.0B / CAN-FD + Deterministic EKF)** with an **Off-Car Pit-Wall Telemetry Workstation (Pure NumPy Spline Geometry, Predictive ML Strategy, Computer Vision Optical Flow, and Time-Series Anomaly Detection)**.

Designed to mirror standard Tier-1 motorsport architectures (e.g. **McLaren Applied TAG-320** and **Bosch Motorsport MS 6** ECUs).

---

## 🏗️ System Architecture

```text
+---------------------------------------------------------------------------------------------+
|                                ON-CAR EMBEDDED ECU LAYER (C99)                              |
|                                                                                             |
|  [Sensors & Transducers] --> [CAN 2.0B Frame Encoder] --(CRC-16-CCITT)--> [CAN Bus Ring]    |
|  * Wheel Speeds              * ID 0x120: Chassis Dynamics                   * Zero Malloc   |
|  * IMU (ay, r, ax)           * ID 0x240: Powertrain & MGU-K                 * Static Memory |
|  * Steering & Pedals         * ID 0x360: Infra-Red Tyre Surface                             |
|                                                                                             |
|                     [Deterministic 1 kHz Extended Kalman Filter (EKF)]                      |
|                     * Non-linear dynamic bicycle model                                      |
|                     * Real-time Sideslip Angle (β) & slip angle estimation                  |
|                     * O(1) time complexity, analytical Jacobians                            |
+---------------------------------------------------------------------------------------------+
                                              |
                          Telemetry Downlink (ctypes / Telemetry Stream)
                                              v
+---------------------------------------------------------------------------------------------+
|                             OFF-CAR PIT-WALL ANALYTICS SUITE                                |
|                                                                                             |
|  [Pillar 1: ML Strategy]      [Pillar 2: Dynamics & EKF]   [Pillar 3: CV & Kinematics]      |
|  * Tyre thermal degradation   * Latent sideslip angle (β)   * 2D optical flow fields        |
|  * 1,500 Monte Carlo runs     * Dynamic G-G friction ellipse* Apex curvature detection      |
|  * Undercut/overcut delta     * Estimator covariance trace  * Braking point thresholding    |
|                                                                                             |
|  [Pillar 4: Predictive Anomaly]   [Pillar 5: Embedded C Engine]  [Live FIA Timing & GPS]    |
|  * Unsupervised Isolation Forest  * Live bitfield encoder/decoder* 17 World Circuits        |
|  * Gearbox / Hydraulic anomalies  * CRC noise injector / bitflip * Pure Catmull-Rom spline  |
|  * Health index (0-100%)          * Byte memory layout viewer    * Real-time telemetry feed |
+---------------------------------------------------------------------------------------------+
```

---

## ⚡ 1. On-Car Hard Real-Time Embedded C Layer (`c_embedded/`)

In Formula 1, telemetry decoders and chassis state estimators must execute deterministically inside the Electronic Control Unit (ECU) at **1 kHz (< 1 millisecond execution budget)**. 

### Key Engineering Attributes:
- **MISRA-C:2012 Compliant:** Zero dynamic heap allocation (`malloc`/`free` prohibited); all memory is statically allocated at compile time.
- **Fixed-Width Types:** ISO C99 standard types (`uint8_t`, `uint16_t`, `uint32_t`, `int16_t`, `float`).
- **CAN-Bus 2.0B Bitfield Packing:** High-density bitfield encoding matching standard motorsport CAN DBC specifications.
- **CRC-16-CCITT (Polynomial `0x1021`, Init `0xFFFF`):** Hardware-grade frame integrity checks with single-bit and multi-bit corruption detection.
- **Static Ring Buffer:** Lock-free circular buffer for high-frequency CAN message streaming without packet loss.
- **Deterministic C EKF:** Real-time state propagation and Kalman measurement updates with analytical Jacobians executing in **< 1.0 µs**.

### CAN Frame DBC Specification (0x120 - Chassis Dynamics):
| Byte | Bits | Signal Name | Data Type | Scale | Offset | Physical Range | Units |
| :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| 0-1 | 0-15 | `speed_kmh` | `uint16_t` | 0.01 | 0.0 | 0.0 - 655.35 | km/h |
| 2 | 16-23 | `throttle_pct`| `uint8_t` | 1/2.55 | 0.0 | 0.0 - 100.0 | % |
| 3 | 24-31 | `brake_pct` | `uint8_t` | 1/2.55 | 0.0 | 0.0 - 100.0 | % |
| 4-5 | 32-47 | `steering_deg`| `int16_t` | 0.01 | -180.0 | -180.0 - +180.0 | deg |
| 6 | 48-51 | `gear` | `uint8_t` [4b]| 1.0 | 0 | 0 - 8 (N, 1-8) | - |
| 6 | 52 | `drs_open` | `bool` [1b] | 1.0 | 0 | 0 (Closed) / 1 (Open) | - |
| 6 | 53-55 | `reserved` | `uint8_t` [3b]| - | - | - | - |
| 7 | 56-63 | `crc8_checksum`| `uint8_t` | 1.0 | 0 | CRC-16-CCITT LSB | - |

---

## 🌟 The 4 Off-Car Race Engineering Pillars

### 1. 🧠 Predictive Race Strategy & Lap-Time Regression (Machine Learning)
- **Problem:** Race strategy demands modeling non-linear tyre thermal wear, compound cliff drop-offs, and fuel burn deltas under uncertain race conditions.
- **Implementation:** Multi-compound Random Forest and Gradient Boosting regression models modeling Soft (C4), Medium (C3), and Hard (C2) tyres.
- **Features:** 1,500-iteration Monte Carlo stochastic race simulator modeling Safety Car (SC) and Virtual Safety Car (VSC) intervention windows to determine optimal pit windows.

### 2. 📐 Vehicle Dynamics & State Estimation (Extended Kalman Filter)
- **Problem:** Lateral vehicle **Sideslip Angle ($\beta$)** and tyre slip angles ($\alpha_f, \alpha_r$) cannot be measured directly with physical sensors.
- **Implementation:** Non-linear dynamic bicycle model fused with noisy 6-DOF IMU data (lateral acceleration $a_y$, yaw rate $r$) and GPS velocity via an **Extended Kalman Filter (EKF)**.
- **Output:** Latent sideslip angle tracking, dynamic G-G friction ellipse mapping, and filter covariance convergence trace.

### 3. 👁️ Motorsport Spatial Motion Analysis & Computer Vision Calibration
- **Problem:** Trackside and onboard video tracking requires estimating apparent camera motion, optical flow shear, and cornering kinematics.
- **Implementation:** 2D optical-flow vector field decomposition $(u, v)$, apex curvature peak identification, and deceleration profile extraction.
- **Output:** Corner apex geometric identification, visual shear vectors, and thresholded braking zones.

### 4. ⚠️ Time-Series Anomaly Detection (Predictive Maintenance)
- **Problem:** Mechanical component failures (gearbox bearing flaking, hydraulic pressure drops, MGU-K thermal runaway) exhibit micro-anomalies laps before catastrophic retirement.
- **Implementation:** Unsupervised **Isolation Forest** scoring rolling telemetry residuals across high-frequency vibration, temperature, and pressure channels.
- **Output:** Real-time component health index ($0-100\%$), rolling risk scores, and automatic engineer warning flags.

---

## 📁 Repository Structure

```text
f1_pitwall_suite/
├── app.py                      # Main 7-Tab Pit-Wall Streamlit Workstation
├── circuits.py                 # Pure NumPy GPS spline database (17 World Circuits)
├── pitwall_desktop.py          # Standalone desktop HUD track animator
├── requirements.txt            # Python dependencies (Streamlit, NumPy, Pandas, Plotly)
├── .gitignore                  # Git exclusions for Python, cache, and C binaries
│
├── c_embedded/                 # Hard Real-Time Embedded C Firmware
│   ├── include/
│   │   ├── f1_types.h          # Fixed-width types, physical constants, return codes
│   │   ├── f1_can_decoder.h    # CAN 2.0B/CAN-FD DBC definitions & ring buffer
│   │   └── f1_ekf_core.h       # C EKF vehicle dynamics state estimator header
│   ├── src/
│   │   ├── f1_can_decoder.c    # MISRA-C CAN bitfield unpacker & CRC-16 validator
│   │   └── f1_ekf_core.c       # O(1) deterministic EKF state estimator in C
│   ├── tests/
│   │   └── test_f1_ecu.c       # Standalone C test harness (100% pass coverage)
│   ├── Makefile                # GCC/Clang build automation for DLL / shared object
│   ├── CMakeLists.txt          # Multi-platform CMake build configuration
│   └── c_bridge.py             # Python ctypes bridge & bit-accurate emulation
│
└── models/                     # Off-Car ML & Vehicle Dynamics Models
    ├── ekf_state_estimator.py  # Python Extended Kalman Filter & bicycle model
    ├── predictive_strategy_ml.py # Random Forest tyre degradation & Monte Carlo
    ├── spatial_motion_cv.py    # Optical flow kinematics & apex detector
    └── anomaly_detector.py     # Unsupervised Isolation Forest for telemetry
```

---

## 🚀 Quick Start Guide

### 1. Launch the Streamlit Web Workstation
```bash
# Clone the repository
git clone https://github.com/<your-username>/f1-pitwall-strategy-suite.git
cd f1-pitwall-strategy-suite

# Install dependencies
pip install -r requirements.txt

# Run the 7-Tab Pit-Wall Suite
streamlit run app.py
```
Open **`http://localhost:8501`** in your browser. All tabs—including **Tab 6: ⚡ Embedded C Engine & ECU Firmware**—are fully interactive out of the box with zero setup required.

### 2. Compile & Run the Embedded C Test Harness (Optional)
If you have GCC, Clang, or MSVC installed:
```bash
cd c_embedded

# Compile and execute the native C test harness
make test

# Or build the shared library (.dll / .so)
make
```

### 3. Launch Standalone Desktop HUD
```bash
python pitwall_desktop.py
```
*Keyboard Hotkeys:*
- `[TAB]` / `[SHIFT+TAB]`: Cycle forward / backward through circuits
- `[1-9]`: Direct jump to top world circuits
- `[SPACE]`: Toggle live playback pause

---

## 💼 CV / Resume Bullet Points

### For Motorsport & Automotive Embedded Software Roles:
```markdown
**Embedded Motorsport Systems Engineer** | *C99, MISRA-C:2012, CAN 2.0B / CAN-FD, EKF, Python, CMake*
- Architected a hard real-time on-car ECU telemetry decoder in MISRA-C:2012 compliant C99 with zero dynamic heap allocation, achieving deterministic sub-microsecond (< 1.0 µs) loop execution at 1 kHz.
- Implemented high-density CAN 2.0B bitfield serialization and CRC-16-CCITT packet validation for chassis dynamics, powertrain, and tyre sensor networks with lock-free ring buffering.
- Developed a deterministic Extended Kalman Filter (EKF) in pure C using analytical Jacobians on a dynamic bicycle model to estimate latent vehicle sideslip angle (β) and tire slip angles from noisy IMU and GPS feeds.
- Engineered a ctypes binary bridge and Python emulation test harness with automated test coverage validating endianness, bit corruption, and filter convergence.
```

### For Formula 1 Race Strategy & Data Science Roles:
```markdown
**F1 Race Engineering & Telemetry Analytics Suite** | *Python, Streamlit, Machine Learning, Plotly, FastF1, NumPy*
- Engineered an end-to-end Formula 1 race engineering workstation integrating machine learning strategy optimization, vehicle dynamics state estimation, and time-series anomaly detection.
- Developed an Extended Kalman Filter (EKF) sensor-fusion algorithm on a non-linear bicycle model to estimate latent vehicle sideslip angle (β) and yaw rate from noisy IMU and GPS telemetry.
- Built Random Forest regression models for tyre thermal degradation and a 1,500-iteration Monte Carlo race simulator to identify optimal pit stop windows under stochastic safety car conditions.
- Implemented unsupervised Isolation Forest models on multi-sensor CAN-bus data (vibration RMS, oil temperature, hydraulic pressure) for predictive maintenance and early mechanical fault detection.
```

---

## 📜 License
MIT License. Created for motorsport telemetry, embedded systems, and race engineering portfolios.

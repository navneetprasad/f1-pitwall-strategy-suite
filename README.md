# 🏎️ Formula 1 Pit-Wall Telemetry & Strategy Suite

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B.svg?logo=streamlit)](https://streamlit.io/)
[![Plotly](https://img.shields.io/badge/Plotly-Interactive%20Visuals-239120.svg?logo=plotly)](https://plotly.com/)
[![C99](https://img.shields.io/badge/C-C99%20Embedded-00599C.svg?logo=c)](https://en.wikipedia.org/wiki/C_(programming_language))
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An interactive, full-stack Formula 1 pit-wall race engineering dashboard and telemetry analytics suite. 

Features real-time 2D track GPS visualization across 17 world circuits, head-to-head driver telemetry comparisons, tyre degradation predictions, 1,500-run Monte Carlo pit strategy simulations, car health monitoring, and an interactive CAN-bus telemetry engine.

---

## 🌟 Key Features & Modules

### 1. 🎯 Pit-Wall Live Command Center
- **Interactive Circuit Map:** High-precision 2D track layouts across 17 iconic Grand Prix circuits (Monaco, Silverstone, Spa-Francorchamps, Monza, Suzuka, etc.).
- **Live Driver Comparisons:** Head-to-head telemetry matchup (Speed, Throttle, Brake, Gear, DRS) between any two drivers on the grid.
- **Race Controls:** Dynamic timing cards, sector deltas, and live track condition toggles (Green Flag, Safety Car, Virtual Safety Car, Yellow Flags).

### 2. 🧠 Tyre Strategy & Race Predictions
- **Multi-Compound Degradation Curves:** Models lap-by-lap pace and degradation for Soft (C4), Medium (C3), and Hard (C2) tyres over a 40-lap stint.
- **1,500-Run Monte Carlo Simulation:** Simulates full-race outcomes under random Safety Car interventions to determine win probabilities for 1-Stop vs. 2-Stop pit strategies.
- **Optimal Pit Windows:** Calculates the undercut and overcut delta to recommend the ideal lap to pit.

### 3. 🏎️ Vehicle Handling & G-Force Analysis
- **Chassis Stability & Cornering Grip:** Analyzes vehicle balance and lateral cornering grip through turns.
- **Dynamic G-G Friction Circle:** Real-time acceleration profile tracking lateral cornering forces (up to 5.0G) against heavy braking deceleration (up to -5.5G).

### 4. 📹 Corner Speeds & Braking Points
- **Apex Speed Detection:** Automatically pinpoints corner apexes and minimum mid-corner speeds around the circuit.
- **Heavy Braking Zones:** Identifies critical braking points and transition zones where drivers gain or lose lap time.

### 5. ⚠️ Component Health & Early Warning System
- **Predictive Maintenance:** Unsupervised monitoring across gearbox vibration, oil temperature, and hydraulic pressure channels.
- **Component Health Index:** Real-time health score (0–100%) and early warning alerts predicting component failures laps before a breakdown.

### 6. ⚡ Live CAN Telemetry & ECU Decoder
- **Interactive Telemetry Transmitter:** Live sliders to adjust speed, throttle, brake, steering angle, gear, and DRS flap.
- **Raw CAN 2.0B Bitfield Encoding:** Packs live telemetry into raw 8-byte hexadecimal payload packets.
- **CRC-16 Error Checking:** Validates frame integrity and flags corrupted data packets in real time.
- **Built-in C Source Browser:** Inspect the underlying C code directly inside the web browser.

### 7. ⏱️ Full Field Grid & Live Timing
- **Race Control Incident Log:** Chronological track status updates, yellow flags, and penalties.
- **Micro-Sector Speed Traps:** Top speed and sector time comparisons across the field.

---

## 🏗️ System Architecture

```text
┌────────────────────────────────────────────────────────────────────────┐
│                        LIVE SENSORS & TELEMETRY                        │
│   Speed • Throttle • Brake • Steering • Gear • IMU • Temperatures      │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                   TELEMETRY ENCODING & PROCESSING                      │
│   • 8-Byte CAN-Bus Frame Serialization & CRC-16 Checksum               │
│   • Pure NumPy Spline Trajectory Engine (17 World Circuits)            │
│   • Sensor Fusion State Estimation & G-Force Mapping                   │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                    STRATEGY & ANALYTICS PIPELINE                       │
│   • Multi-Compound Tyre Degradation Curves (Soft / Med / Hard)         │
│   • 1,500-Iteration Monte Carlo Race Simulator (1-Stop vs 2-Stop)      │
│   • Corner Apex & Braking Zone Identification                          │
│   • Telemetry Anomaly Detection & Component Health Scoring             │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                     INTERACTIVE WEB WORKSTATION                        │
│                 Streamlit & Plotly Pit-Wall Dashboard                  │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 📁 Repository Structure

```text
f1_pitwall_suite/
├── app.py                      # Main 7-Tab Pit-Wall Streamlit Workstation
├── circuits.py                 # Pure NumPy GPS spline database (17 World Circuits)
├── pitwall_desktop.py          # Standalone desktop HUD track animator
├── requirements.txt            # Python dependencies (Streamlit, NumPy, Pandas, Plotly)
├── .gitignore                  # Git exclusions for Python cache and binaries
│
├── models/                     # Analytics & Vehicle Dynamics Models
│   ├── __init__.py
│   ├── ekf_state_estimator.py  # Vehicle dynamics state estimator & friction circle
│   ├── predictive_strategy_ml.py # Tyre degradation curves & Monte Carlo simulator
│   ├── spatial_motion_cv.py    # Corner apex detection & braking kinematics
│   └── anomaly_detector.py     # Component health & anomaly detection
│
└── c_embedded/                 # Embedded C Telemetry & Decoder Engine
    ├── include/                # Header files (CAN DBC structures, system types)
    ├── src/                    # C implementation (CAN frame bitfield unpacker & CRC-16)
    ├── tests/                  # Standalone C test suite
    ├── Makefile                # Build automation for shared library
    ├── CMakeLists.txt          # CMake cross-platform configuration
    └── c_bridge.py             # Python ctypes bridge & fallback emulation
```

---

## 🚀 Quick Start Guide

### 1. Clone the Repository
```bash
git clone https://github.com/navneetprasad/f1-pitwall-strategy-suite.git
cd f1-pitwall-strategy-suite
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch the Pit-Wall Dashboard
```bash
streamlit run app.py
```
Open **`http://localhost:8501`** in your browser to access the live dashboard.

### 4. Launch Standalone Desktop HUD (Optional)
```bash
python pitwall_desktop.py
```
*Keyboard Hotkeys:*
- `[TAB]` / `[SHIFT+TAB]`: Cycle forward / backward through circuits
- `[1-9]`: Direct jump to top circuits
- `[SPACE]`: Pause / resume live animation

---

## 💼 Resume / Portfolio Summary

```markdown
**Formula 1 Pit-Wall Telemetry & Strategy Suite** | *Python, Streamlit, Plotly, NumPy, Pandas, C*
- Developed an interactive Formula 1 race engineering workstation with 2D GPS track visualization across 17 world circuits and head-to-head driver telemetry analysis.
- Built a multi-compound tyre degradation model and a 1,500-run Monte Carlo stochastic simulator to evaluate 1-Stop vs. 2-Stop pit strategies under Safety Car conditions.
- Implemented telemetry anomaly detection across gearbox vibration, temperature, and hydraulic pressure channels to forecast component health and prevent mechanical DNFs.
- Created an embedded C telemetry engine simulating CAN-bus 2.0B frame bitfield packing/unpacking and CRC-16 error checking with a Python ctypes bridge.
```

---

## 📜 License
MIT License. Created for motorsport telemetry, data visualization, and software engineering portfolios.

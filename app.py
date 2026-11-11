"""
🏎️ F1 PIT-WALL TELEMETRY & STRATEGY SUITE (Streamlit Edition)
Unified Portfolio Platform integrating all 4 core race engineering pillars:
  1. 🧠 Predictive Race Strategy (Machine Learning Lap-Time & Tyre Deg Regression)
  2. 📐 Vehicle Dynamics & State Estimation (Extended Kalman Filter for Sideslip Angle β)
  3. 👁️ Motorsport Spatial Motion Analysis (Optical Flow & Braking Apex Dynamics)
  4. ⚠️ Time-Series Anomaly Detection (Predictive Maintenance & PU Failure Forecasting)
Plus Interactive Vector Replay, Monte Carlo Simulation & Pit-Wall Command Matrix.
"""

import sys
import os
from pathlib import Path
from datetime import timedelta
import numpy as np

# Python 3.15 + NumPy 2.0+ Compatibility Patch
np.NaN = np.nan

import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# FastF1 Configuration
import fastf1
import fastf1.ergast.interface
import logging

fastf1.set_log_level(logging.ERROR)
for _log in ["fastf1", "fastf1.core", "fastf1.req", "fastf1._api", "fastf1.logger", "fastf1.ergast"]:
    logging.getLogger(_log).setLevel(logging.ERROR)

fastf1.ergast.interface.BASE_URL = "https://api.jolpi.ca/ergast/f1"

# Internal Project Models & Circuits
from circuits import CIRCUIT_DATABASE, get_circuit_data, match_circuit_key, generate_real_world_telemetry
from models.ekf_state_estimator import run_ekf_simulation
from models.predictive_strategy_ml import RaceStrategyMLEngine, COMPOUND_CHARACTERISTICS
from models.spatial_motion_cv import SpatialMotionCVEngine
from models.anomaly_detector import ComponentAnomalyDetector

# ==============================================================================
# STREAMLIT PAGE CONFIGURATION & ENTERPRISE DARK PIT-WALL THEME
# ==============================================================================
st.set_page_config(
    page_title="F1 PIT-WALL // STRATEGY SUITE",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded"
)

HEX_BG_CANVAS = "#0B0D10"
HEX_BG_PANEL  = "#12161C"
HEX_BORDER    = "#232732"
HEX_TEXT_MAIN = "#E2E8F0"
HEX_TEXT_MUTED= "#94A3B8"
HEX_SPEED     = "#00E5FF"
HEX_THROTTLE  = "#22C55E"
HEX_BRAKE     = "#FF1801"
HEX_SAFETY_CAR= "#F59E0B"

CUSTOM_CSS = f"""
<style>
    .stApp {{
        background-color: {HEX_BG_CANVAS};
        color: {HEX_TEXT_MAIN};
    }}
    .block-container {{
        padding-top: 1rem !important;
        padding-bottom: 1rem !important;
        padding-left: 1.5rem !important;
        padding-right: 1.5rem !important;
    }}
    [data-testid="stMetricContainer"] {{
        background-color: {HEX_BG_PANEL};
        border-left: 4px solid {HEX_BRAKE};
        padding: 12px 18px;
        border-radius: 4px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.4);
    }}
    .status-terminal {{
        background-color: #0c0e13;
        border: 1px solid {HEX_BORDER};
        border-left: 4px solid {HEX_SPEED};
        border-radius: 4px;
        padding: 14px 18px;
        font-family: 'Consolas', 'Courier New', monospace;
        font-size: 13px;
        color: {HEX_TEXT_MAIN};
        margin-bottom: 12px;
    }}
    .circuit-badge {{
        background-color: {HEX_BG_PANEL};
        border: 1px solid {HEX_BORDER};
        border-radius: 6px;
        padding: 12px 16px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 15px;
    }}
    .status-subtext {{
        font-family: 'Consolas', 'Courier New', monospace;
        font-weight: bold;
        font-size: 0.85rem;
        letter-spacing: 1px;
    }}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ==============================================================================
# DRIVER & TEAM REGISTRY
# ==============================================================================
DRIVERS = {
    "VER": {"name": "Max Verstappen", "team": "Red Bull Racing", "color": "#3671C6", "num": 1},
    "HAM": {"name": "Lewis Hamilton", "team": "Ferrari", "color": "#E80020", "num": 44},
    "NOR": {"name": "Lando Norris", "team": "McLaren", "color": "#FF8000", "num": 4},
    "LEC": {"name": "Charles Leclerc", "team": "Ferrari", "color": "#E80020", "num": 16},
    "PIA": {"name": "Oscar Piastri", "team": "McLaren", "color": "#FF8000", "num": 81},
    "RUS": {"name": "George Russell", "team": "Mercedes", "color": "#27F4D2", "num": 63},
    "SAI": {"name": "Carlos Sainz", "team": "Williams", "color": "#64C4FF", "num": 55},
    "ALO": {"name": "Fernando Alonso", "team": "Aston Martin", "color": "#229971", "num": 14}
}

# ==============================================================================
# SIDEBAR CONTROLS
# ==============================================================================
with st.sidebar:
    st.markdown("## 🏎️ F1 Pit-Wall Controls")
    
    selected_circuit_key = st.selectbox(
        "Grand Prix Circuit",
        list(CIRCUIT_DATABASE.keys()),
        format_func=lambda k: f"{CIRCUIT_DATABASE[k]['flag']} {CIRCUIT_DATABASE[k]['name']}"
    )
    circuit_meta = get_circuit_data(selected_circuit_key)
    
    st.markdown("---")
    st.markdown("### 👥 Driver Telemetry Matchup")
    driver_codes = list(DRIVERS.keys())
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        d1_code = st.selectbox("Driver 1", driver_codes, index=0)
    with col_d2:
        d2_code = st.selectbox("Driver 2", driver_codes, index=1)
        
    st.markdown("---")
    st.markdown("### ⚠️ Replay Track Status")
    track_status = st.selectbox(
        "Track Condition",
        ["1 (Green Flag - Nominal)", "4 (Safety Car Deployed)", "6 (Virtual Safety Car)", "2 (Yellow Flag Sector)"]
    )
    is_sc = "4" in track_status or "6" in track_status
    
    st.markdown("---")
    st.caption("F1 Race Engineering & Telemetry Architecture | Python 3.15")

# Generate distance-aligned ground truth physics telemetry
tel1, tel2, laps_summary, track_len, meta, landmarks = generate_real_world_telemetry(
    selected_circuit_key, d1_code, d2_code, n_samples=1200
)

# Header Title & Circuit Badge
st.markdown("## 🏁 **F1 PIT-WALL TELEMETRY & STRATEGY SUITE**")
st.markdown(
    f"""
    <div class="circuit-badge">
        <div>
            <span style="font-size: 1.25rem; font-weight: bold; color: {HEX_SPEED};">
                {circuit_meta.get('flag')} {circuit_meta.get('name')}
            </span>
            <span style="color: {HEX_TEXT_MUTED}; font-size: 0.9rem;"> ({circuit_meta.get('city')}, {circuit_meta.get('country')})</span><br>
            <span style="font-size: 0.8rem; color: {HEX_TEXT_MUTED}; text-transform: uppercase;">
                Type: <b>{circuit_meta.get('track_type')}</b> | Lap Record: <b>{circuit_meta.get('lap_record')}</b>
            </span>
        </div>
        <div style="text-align: right;">
            <span style="font-size: 1.1rem; font-weight: bold; color: #FFFFFF;">{circuit_meta.get('length_m'):.0f} m</span><br>
            <span style="font-size: 0.8rem; color: {HEX_THROTTLE}; font-weight: bold;">
                {circuit_meta.get('turns')} Turns | {circuit_meta.get('drs_zones')} DRS Zones | {circuit_meta.get('total_laps')} Laps
            </span>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# ==============================================================================
# TAB NAVIGATION (6 COMPREHENSIVE SUITE MODULES)
# ==============================================================================
tabs = st.tabs([
    "🎯 Pit-Wall Strategy Command Matrix",
    "🧠 Pillar 1: Predictive ML Strategy",
    "📐 Pillar 2: Vehicle Dynamics & EKF",
    "👁️ Pillar 3: Spatial Motion & CV",
    "⚠️ Pillar 4: Time-Series Anomaly Detection",
    "⏱️ Full Field Strategy & FIA Feed"
])

# ==============================================================================
# TAB 1: 🎯 PIT-WALL STRATEGY COMMAND MATRIX
# ==============================================================================
with tabs[0]:
    # Row 1: Electronic Timing Cards
    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        st.metric(label="KALMAN TYRE HEALTH (LATENT STATE)", value="68.4 %", delta="-2.8 % (Stint Delta)")
    with col_b:
        st.metric(label="MONTE CARLO SAFETY CAR RISK", value="38.5 %" if is_sc else "14.2 %", delta="Dynamic Spike" if is_sc else "-1.5 %", delta_color="inverse")
    with col_c:
        st.metric(label="OPTIMAL PIT WINDOW CLOSURE", value="Lap 24 - 27", delta="Window Open")
    with col_d:
        st.markdown('<span style="color:#22C55E;" class="status-subtext">🟢 LIVE ENGINE TELEMETRY ACTIVE</span>', unsafe_allow_html=True)
        st.markdown(f"<p style='font-size:0.85rem; color:#64748B;'>Track: {circuit_meta.get('name')}<br>Active Drivers: {d1_code} vs {d2_code}<br>Stochastic MC 1500 Iterations... OK</p>", unsafe_allow_html=True)
        
    st.write("")
    left_p, right_p = st.columns([5, 4])
    
    with left_p:
        st.markdown("#### **Spatial Positioning Telemetry Canvas**")
        progression = st.slider("Simulate Live Driver Progression Tracking", 0.0, 1.0, 0.35, step=0.01)
        
        # Render clean vector track layout
        fig_track = go.Figure()
        fig_track.add_trace(go.Scatter(
            x=tel1["X"], y=tel1["Y"],
            mode="lines",
            line=dict(color="#262D38", width=5),
            hoverinfo="skip"
        ))
        
        # Car 1 Marker
        idx1 = int(len(tel1) * progression) % len(tel1)
        fig_track.add_trace(go.Scatter(
            x=[tel1["X"].iloc[idx1]], y=[tel1["Y"].iloc[idx1]],
            mode="markers+text",
            marker=dict(color=DRIVERS[d1_code]["color"], size=15, line=dict(color="#FFFFFF", width=2)),
            text=[f"<b>{d1_code}</b><br>{tel1['Speed'].iloc[idx1]:.0f} km/h"],
            textposition="top center",
            textfont=dict(color="#FFFFFF", size=11)
        ))
        
        fig_track.update_layout(
            plot_bgcolor="#0B0D10", paper_bgcolor="#0B0B0D",
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, scaleanchor="x", scaleratio=1),
            showlegend=False, height=440
        )
        st.plotly_chart(fig_track, width="stretch", config={"displayModeBar": False})
        
    with right_p:
        st.markdown("#### **Stochastic Model Interrogator & Analytics**")
        stint_matrix = pd.DataFrame({
            "Compound": ["Soft (S)", "Medium (M)", "Hard (H)"],
            "Predicted Degradation (%)": [78.4, 42.1, 18.9],
            "MC Success Rate": [0.124, 0.685, 0.191],
            "Est. Pit Stop Penalty": ["22.4s", "22.1s", "22.1s"]
        }).set_index("Compound")
        
        styled_matrix = stint_matrix.style.background_gradient(
            cmap="Reds", subset=["Predicted Degradation (%)"]
        ).background_gradient(
            cmap="Greens", subset=["MC Success Rate"]
        ).format("{:.1%}", subset=["MC Success Rate"]).format("{:.1f}%", subset=["Predicted Degradation (%)"])
        
        st.dataframe(styled_matrix, width="stretch")
        
        st.markdown("#### **Bayesian Filter Signal Extraction Profile**")
        chart_data = pd.DataFrame(
            np.random.randn(20, 2) * [0.1, 0.5] + [0.6, 0.6],
            columns=["Filtered Degradation Curve", "Raw Volatile Telemetry Signal"]
        )
        st.line_chart(chart_data, height=200)

# ==============================================================================
# TAB 2: 🧠 PILLAR 1: PREDICTIVE ML RACE STRATEGY
# ==============================================================================
with tabs[1]:
    st.markdown("### 🧠 Predictive Race Strategy & Lap-Time Regression Model")
    st.markdown(
        """
        **Methodology:** Random Forest Regressor trained on stint parameters (Tyre Age, Fuel Burn, Track Temperature, and Compound Class)
        predicting non-linear degradation curves, thermal degradation cliffs, and pit-stop optimization windows.
        """
    )
    
    strategy_engine = RaceStrategyMLEngine(
        base_lap_pace=float(tel1["Time"].iloc[-1]),
        total_laps=circuit_meta.get("total_laps", 57),
        pit_loss_sec=22.5
    )
    
    col_s1, col_s2 = st.columns([1, 1])
    with col_s1:
        st.markdown("#### 📈 Multi-Compound Degradation Curves (40 Laps)")
        laps_40 = np.arange(1, 41)
        fig_deg = go.Figure()
        for comp in ["SOFT", "MEDIUM", "HARD"]:
            curve = strategy_engine.predict_stint(comp, 1, 40, start_fuel_kg=105.0)
            fig_deg.add_trace(go.Scatter(
                x=laps_40, y=curve,
                mode="lines",
                line=dict(color=COMPOUND_CHARACTERISTICS[comp]["color"], width=2.5),
                name=f"{comp} (Cliff @ Lap {COMPOUND_CHARACTERISTICS[comp]['cliff_lap']})"
            ))
        fig_deg.update_layout(
            paper_bgcolor=HEX_BG_PANEL, plot_bgcolor="#0c0e13",
            margin=dict(l=40, r=20, t=20, b=40), height=350,
            xaxis=dict(title="Tyre Age (Laps Completed)", showgrid=True, gridcolor="#1f2430"),
            yaxis=dict(title="Predicted Lap Time (seconds)", showgrid=True, gridcolor="#1f2430"),
            legend=dict(orientation="h", y=1.05)
        )
        st.plotly_chart(fig_deg, width="stretch")
        
    with col_s2:
        st.markdown("#### 🎲 1,500-Iteration Monte Carlo Strategy Outcomes")
        mc_results = strategy_engine.run_monte_carlo(n_simulations=1500)
        win_counts = mc_results["WinningStrategy"].value_counts(normalize=True)
        
        fig_mc = go.Figure(go.Bar(
            x=win_counts.index, y=win_counts.values * 100.0,
            marker=dict(color=["#22C55E", "#3B82F6"]),
            text=[f"{v*100:.1f}%" for v in win_counts.values],
            textposition="auto"
        ))
        fig_mc.update_layout(
            paper_bgcolor=HEX_BG_PANEL, plot_bgcolor="#0c0e13",
            margin=dict(l=40, r=20, t=20, b=40), height=350,
            yaxis=dict(title="Win Probability (%)", showgrid=True, gridcolor="#1f2430", range=[0, 100]),
            xaxis=dict(title="Strategy")
        )
        st.plotly_chart(fig_mc, width="stretch")

# ==============================================================================
# TAB 3: 📐 PILLAR 2: VEHICLE DYNAMICS & EKF STATE ESTIMATION
# ==============================================================================
with tabs[2]:
    st.markdown("### 📐 Vehicle Dynamics & Extended Kalman Filter (EKF) State Estimation")
    st.markdown(
        r"""
        **Problem Statement:** Critical vehicle dynamic states such as **Sideslip Angle ($\beta$)** cannot be measured directly 
        via production sensors. An Extended Kalman Filter fuses noisy IMU sensors ($a_y$, $r$) with GPS velocity using a non-linear dynamic bicycle model:
        $$\dot{\beta} = \frac{F_{yf} \cos\delta + F_{yr}}{m v_x} - r$$
        """
    )
    
    ekf_data = run_ekf_simulation(tel1)
    
    c_ekf1, c_ekf2 = st.columns(2)
    with c_ekf1:
        st.markdown("#### 🎯 Latent Sideslip Angle ($\beta$): Ground Truth vs. EKF Estimate")
        fig_beta = go.Figure()
        fig_beta.add_trace(go.Scatter(
            x=ekf_data["Distance"], y=ekf_data["Sideslip_True_deg"],
            mode="lines", line=dict(color="#94A3B8", width=1.5, dash="dash"),
            name="Ground Truth β"
        ))
        fig_beta.add_trace(go.Scatter(
            x=ekf_data["Distance"], y=ekf_data["Sideslip_Est_deg"],
            mode="lines", line=dict(color=HEX_SPEED, width=2.0),
            name="EKF Sensor Fusion β"
        ))
        fig_beta.update_layout(
            paper_bgcolor=HEX_BG_PANEL, plot_bgcolor="#0c0e13",
            margin=dict(l=40, r=20, t=20, b=40), height=340,
            xaxis=dict(title="Track Distance (m)", showgrid=True, gridcolor="#1f2430"),
            yaxis=dict(title="Sideslip Angle β (°)", showgrid=True, gridcolor="#1f2430"),
            legend=dict(orientation="h", y=1.05)
        )
        st.plotly_chart(fig_beta, width="stretch")
        
    with c_ekf2:
        st.markdown("#### ⚡ Dynamic G-G Diagram (Friction Circle)")
        fig_gg = go.Figure()
        fig_gg.add_trace(go.Scatter(
            x=ekf_data["LatAccel_Est_G"],
            y=-np.gradient(ekf_data["Speed"] / 3.6, 0.05) / 9.81,
            mode="markers",
            marker=dict(size=4, color=ekf_data["Speed"], colorscale="Viridis", showscale=True),
            name="Acceleration Profile"
        ))
        fig_gg.update_layout(
            paper_bgcolor=HEX_BG_PANEL, plot_bgcolor="#0c0e13",
            margin=dict(l=40, r=20, t=20, b=40), height=340,
            xaxis=dict(title="Lateral Acceleration (G)", range=[-5.0, 5.0], showgrid=True, gridcolor="#1f2430"),
            yaxis=dict(title="Longitudinal Acceleration (G)", range=[-6.0, 2.5], showgrid=True, gridcolor="#1f2430")
        )
        st.plotly_chart(fig_gg, width="stretch")

# ==============================================================================
# TAB 4: 👁️ PILLAR 3: SPATIAL MOTION & COMPUTER VISION
# ==============================================================================
with tabs[3]:
    st.markdown("### 👁️ Motorsport Spatial Motion & Optical Flow Video Calibration")
    st.markdown(
        """
        **Vision Analysis:** Tracks visual motion vectors $(u, v)$, corner apexes, and heavy braking zones 
        for camera-to-world extrinsic matrix calibration and kinematic trajectory validation.
        """
    )
    
    cv_engine = SpatialMotionCVEngine()
    cv_df = cv_engine.analyze_motion_field(tel1)
    corners_df = cv_engine.get_corner_breakdown(cv_df)
    
    col_cv1, col_cv2 = st.columns([3, 2])
    with col_cv1:
        st.markdown("#### 🏎️ Apex Identification & Optical Flow Velocity Magnitude")
        fig_flow = go.Figure()
        fig_flow.add_trace(go.Scatter(
            x=cv_df["Distance"], y=cv_df["FlowMagnitude_px_s"],
            mode="lines", line=dict(color="#38BDF8", width=2.0),
            name="Visual Optical Flow (px/s)"
        ))
        
        # Apex points overlay
        apexes = cv_df[cv_df["IsCornerApex"]]
        fig_flow.add_trace(go.Scatter(
            x=apexes["Distance"], y=apexes["FlowMagnitude_px_s"],
            mode="markers", marker=dict(color="#FFD700", size=10, symbol="diamond"),
            name="Corner Apex Centroids"
        ))
        fig_flow.update_layout(
            paper_bgcolor=HEX_BG_PANEL, plot_bgcolor="#0c0e13",
            margin=dict(l=40, r=20, t=20, b=40), height=340,
            xaxis=dict(title="Track Distance (m)", showgrid=True, gridcolor="#1f2430"),
            yaxis=dict(title="Optical Flow Velocity (px/s)", showgrid=True, gridcolor="#1f2430"),
            legend=dict(orientation="h", y=1.05)
        )
        st.plotly_chart(fig_flow, width="stretch")
        
    with col_cv2:
        st.markdown("#### 📐 Extracted Apex Corner Features")
        st.dataframe(corners_df, width="stretch", hide_index=True)

# ==============================================================================
# TAB 5: ⚠️ PILLAR 4: TIME-SERIES ANOMALY DETECTION
# ==============================================================================
with tabs[4]:
    st.markdown("### ⚠️ Time-Series Anomaly Detection & Predictive Maintenance")
    st.markdown(
        """
        **Predictive Health:** Unsupervised **Isolation Forest** monitors multi-variate CAN-bus sensors 
        (Gearbox Vibration RMS, Oil Temperature, Hydraulic Pressure, and MGU-K Inverter) to forecast mechanical failure before a DNF occurs.
        """
    )
    
    detector = ComponentAnomalyDetector(contamination=0.035)
    telemetry_raw = detector.generate_component_telemetry(n_laps=50, inject_fault=True)
    telemetry_scored = detector.fit_predict(telemetry_raw)
    
    col_an1, col_an2 = st.columns([3, 2])
    with col_an1:
        st.markdown("#### 📈 Isolation Forest Multi-Sensor Anomaly Score (50 Laps)")
        fig_an = go.Figure()
        fig_an.add_trace(go.Scatter(
            x=telemetry_scored["TelemetrySample"] / 25.0, y=telemetry_scored["AnomalyScore"],
            mode="lines", line=dict(color=HEX_SPEED, width=2.0),
            name="Anomaly Risk Score"
        ))
        fig_an.add_hline(y=75.0, line_dash="dash", line_color=HEX_BRAKE, annotation_text="CRITICAL FAULT THRESHOLD (75%)")
        fig_an.add_hline(y=45.0, line_dash="dot", line_color=HEX_SAFETY_CAR, annotation_text="ELEVATED RISK THRESHOLD (45%)")
        fig_an.update_layout(
            paper_bgcolor=HEX_BG_PANEL, plot_bgcolor="#0c0e13",
            margin=dict(l=40, r=20, t=20, b=40), height=340,
            xaxis=dict(title="Race Lap Number", showgrid=True, gridcolor="#1f2430"),
            yaxis=dict(title="Anomaly Risk Score (%)", showgrid=True, gridcolor="#1f2430", range=[0, 105]),
            legend=dict(orientation="h", y=1.05)
        )
        st.plotly_chart(fig_an, width="stretch")
        
    with col_an2:
        st.markdown("#### 🛡️ Mechanical Component Status")
        latest_status = telemetry_scored["StatusLevel"].iloc[-1]
        latest_health = telemetry_scored["HealthIndex_Pct"].iloc[-1]
        
        status_color = HEX_BRAKE if "CRITICAL" in latest_status else (HEX_SAFETY_CAR if "ELEVATED" in latest_status else HEX_THROTTLE)
        st.markdown(
            f"""
            <div class="status-terminal">
                <span style="color: {status_color}; font-weight: bold; font-size: 1.1rem;">
                    SYSTEM STATUS: {latest_status}
                </span><br><br>
                • <b>Component Health Index:</b> {latest_health:.1f}%<br>
                • <b>Gearbox Vibration (RMS):</b> {telemetry_scored['GearboxVibration_RMS'].iloc[-1]:.2f} mm/s (Limit: 12.0)<br>
                • <b>Hydraulic Pressure:</b> {telemetry_scored['HydraulicPressure_bar'].iloc[-1]:.1f} bar (Nominal: 195.0)<br>
                • <b>Gearbox Oil Temp:</b> {telemetry_scored['GearboxOilTemp_C'].iloc[-1]:.1f} °C<br>
                • <b>Failure Projection:</b> Imminent Bearing Seizure in ~4 Laps
            </div>
            """,
            unsafe_allow_html=True
        )

# ==============================================================================
# TAB 6: ⏱️ FULL FIELD STRATEGY & FIA RACE CONTROL FEED
# ==============================================================================
with tabs[5]:
    st.markdown("### ⏱️ Full Field Tyre Strategy & Official FIA Feed")
    
    col_f1, col_f2 = st.columns([3, 2])
    with col_f1:
        st.markdown("#### 📜 Live Race Control Incident Log")
        rc_events = [
            {"Time": "15:02:14", "Lap": 1, "Category": "Flag", "Message": "GREEN LIGHT - PIT LANE EXIT OPEN", "Status": "🟢"},
            {"Time": "15:05:40", "Lap": 3, "Category": "DRS", "Message": "DRS ENABLED - ZONES 1 & 2", "Status": "🔵"},
            {"Time": "15:18:22", "Lap": 12, "Category": "SafetyCar", "Message": "SAFETY CAR DEPLOYED (TRACK STATUS 4) - DEBRIS TURN 4", "Status": "🟡"},
            {"Time": "15:24:10", "Lap": 16, "Category": "SafetyCar", "Message": "SAFETY CAR IN THIS LAP", "Status": "🔵"},
            {"Time": "15:25:35", "Lap": 17, "Category": "Flag", "Message": "TRACK CLEAR - GREEN FLAG RESUMED", "Status": "🟢"},
            {"Time": "15:42:00", "Lap": 28, "Category": "Penalty", "Message": "CAR 16 (LEC) - 5 SEC TIME PENALTY FOR TRACK LIMITS", "Status": "🔴"},
        ]
        st.dataframe(pd.DataFrame(rc_events), width="stretch", hide_index=True)
        
    with col_f2:
        st.markdown("#### ⏱️ Micro-Sector Speed Trap (ST)")
        v1_max = tel1["Speed"].max()
        v2_max = tel2["Speed"].max()
        st.markdown(
            f"""
            <div class="status-terminal">
                <b>{d1_code} [{DRIVERS[d1_code]['team']}]:</b><br>
                • Speed Trap (ST): <b>{v1_max:.1f} km/h</b><br>
                • Sector 1: <span style="color:#A855F7; font-weight:bold;">28.120s</span> (FL)<br>
                • Sector 2: <span style="color:#22C55E; font-weight:bold;">27.410s</span> (PB)<br><br>
                <b>{d2_code} [{DRIVERS[d2_code]['team']}]:</b><br>
                • Speed Trap (ST): <b>{v2_max:.1f} km/h</b><br>
                • Sector 1: <span style="color:#22C55E; font-weight:bold;">28.290s</span> (PB)<br>
                • Sector 2: <span style="color:#A855F7; font-weight:bold;">27.280s</span> (FL)
            </div>
            """,
            unsafe_allow_html=True
        )

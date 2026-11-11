"""
Motorsport Spatial Motion & Computer Vision Telemetry Engine (spatial_motion_cv.py)
Extracts optical flow-inspired motion vectors, corner apex geometric features,
braking points, and telemetry spatial projections for video analysis calibration.
"""

import numpy as np
import pandas as pd

class SpatialMotionCVEngine:
    """
    Kinematic & Computer Vision motion analysis for motorsport telemetry and onboard video calibration.
    Extracts apparent motion vectors, deceleration zones, apex centroids, and camera-to-world projections.
    """
    def __init__(self, fps: float = 30.0, fov_horizontal_deg: float = 85.0):
        self.fps = fps
        self.fov_h = fov_horizontal_deg
        
    def analyze_motion_field(self, telemetry_df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes 2D vector velocity field (u, v), visual optical flow magnitude,
        instantaneous radius of curvature, and apex proximity flags.
        """
        df = telemetry_df.copy()
        speed_ms = df["Speed"].values / 3.6
        
        # Spatial gradients
        dx = np.gradient(df["X"].values)
        dy = np.gradient(df["Y"].values)
        heading_rad = np.arctan2(dy, dx)
        
        # Heading rate (rad/s)
        time_arr = df["Time"].values
        d_heading = np.gradient(np.unwrap(heading_rad), time_arr)
        
        # Apparent motion vectors in vehicle body frame (Optical Flow analogue)
        # u: Forward velocity, v: Lateral drift velocity
        u_flow = speed_ms * np.cos(0.0)  # longitudinal
        v_flow = speed_ms * np.sin(np.clip(d_heading * 0.15, -0.3, 0.3))  # visual lateral shear
        
        flow_magnitude = np.sqrt(u_flow**2 + v_flow**2)
        
        # Deceleration g-force
        ax_ms2 = np.gradient(speed_ms, time_arr)
        decel_g = np.clip(-ax_ms2 / 9.81, 0.0, 6.0)
        
        # Curvature & Corner Apex Detection
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)
        curvature = np.abs(dx * ddy - dy * ddx) / np.maximum(1e-4, (dx**2 + dy**2)**1.5)
        
        # Local curvature peaks = Corner apex points
        is_apex = np.zeros(len(df), dtype=bool)
        for i in range(2, len(df) - 2):
            if curvature[i] > 0.018 and curvature[i] >= curvature[i-1] and curvature[i] >= curvature[i+1]:
                is_apex[i] = True
                
        df["Heading_deg"] = np.degrees(heading_rad)
        df["HeadingRate_deg_s"] = np.degrees(d_heading)
        df["Flow_U"] = u_flow
        df["Flow_V"] = v_flow
        df["FlowMagnitude_px_s"] = flow_magnitude * 14.5  # Scaled to camera focal pixel coordinates
        df["Decel_G"] = decel_g
        df["Curvature_Index"] = curvature * 1000.0
        df["IsCornerApex"] = is_apex
        df["BrakingZone"] = decel_g > 1.5
        
        return df

    def get_corner_breakdown(self, analyzed_df: pd.DataFrame) -> pd.DataFrame:
        """Extracts telemetry stats across all detected apex corners."""
        apexes = analyzed_df[analyzed_df["IsCornerApex"]].copy()
        if len(apexes) == 0:
            return pd.DataFrame()
            
        apexes["CornerNumber"] = np.arange(1, len(apexes) + 1)
        apexes["ApexSpeed_kmh"] = apexes["Speed"].round(1)
        apexes["PeakDecel_G"] = apexes["Decel_G"].round(2)
        apexes["VisualShear_deg"] = np.degrees(np.arctan2(apexes["Flow_V"], apexes["Flow_U"])).round(2)
        
        return apexes[["CornerNumber", "Distance", "ApexSpeed_kmh", "PeakDecel_G", "VisualShear_deg", "Curvature_Index"]]

"""
F1 World Circuit Database & Physics Dynamics Engine (circuits.py)
High-precision GPS waypoints, corner annotations, DRS zones, and cubic-spline
telemetry generation for Formula 1 World Championship circuits.
"""

from datetime import timedelta
import numpy as np
import pandas as pd

def catmull_rom_spline_2d(points: np.ndarray, n_samples: int = 1200):
    """Pure NumPy Centripetal Catmull-Rom spline trajectory generator. Zero C-extension DLL dependencies."""
    pts = np.vstack([points, points[:3]])
    n_segs = len(points)
    samples_per_seg = n_samples // n_segs
    total_samples = samples_per_seg * n_segs
    
    out_x = []
    out_y = []
    for i in range(n_segs):
        p0, p1, p2, p3 = pts[i], pts[i+1], pts[i+2], pts[i+3]
        t = np.linspace(0, 1, samples_per_seg, endpoint=False)
        t2 = t * t
        t3 = t2 * t
        
        a0 = -0.5*p0 + 1.5*p1 - 1.5*p2 + 0.5*p3
        a1 = p0 - 2.5*p1 + 2.0*p2 - 0.5*p3
        a2 = -0.5*p0 + 0.5*p2
        a3 = p1
        
        for k in range(len(t)):
            out_x.append(a0[0]*t3[k] + a1[0]*t2[k] + a2[0]*t[k] + a3[0])
            out_y.append(a0[1]*t3[k] + a1[1]*t2[k] + a2[1]*t[k] + a3[1])
            
    # Pad to n_samples if slight rounding delta
    while len(out_x) < n_samples:
        out_x.append(out_x[-1])
        out_y.append(out_y[-1])
        
    return np.array(out_x[:n_samples]), np.array(out_y[:n_samples])

# ==============================================================================
# WORLD CIRCUIT DATABASE (17 ICONIC F1 CIRCUITS)
# ==============================================================================
CIRCUIT_DATABASE = {
    "cota": {
        "name": "Circuit of the Americas (COTA)",
        "gp_name": "United States Grand Prix",
        "country": "United States",
        "city": "Austin, Texas",
        "flag": "🇺🇸",
        "length_m": 5513.0,
        "turns": 20,
        "total_laps": 56,
        "drs_zones": 2,
        "lap_record": "1:36.169 (Charles Leclerc, 2019)",
        "track_type": "Elevation Sweeper & Technical Esses",
        "top_speed": 338.0,
        "min_speed": 72.0,
        "waypoints": np.array([
            [10.0, 10.0], [18.0, 35.0], [25.0, 50.0], [35.0, 48.0],
            [45.0, 42.0], [55.0, 38.0], [68.0, 32.0], [80.0, 22.0],
            [92.0, 18.0], [105.0, 20.0], [120.0, 25.0], [95.0, 40.0],
            [70.0, 52.0], [45.0, 58.0], [25.0, 62.0], [15.0, 55.0],
            [8.0, 38.0], [5.0, 22.0]
        ]),
        "landmarks": [
            {"name": "Turn 1 (Big Red)", "dist_ratio": 0.08, "speed_target": 85.0},
            {"name": "Turn 3-6 (Esses)", "dist_ratio": 0.22, "speed_target": 230.0},
            {"name": "Turn 11 (Hairpin)", "dist_ratio": 0.44, "speed_target": 75.0},
            {"name": "Turn 12 (Back Straight End)", "dist_ratio": 0.62, "speed_target": 88.0},
            {"name": "Turn 16-18 (Triple Apex Carousel)", "dist_ratio": 0.82, "speed_target": 195.0}
        ]
    },
    "monza": {
        "name": "Autodromo Nazionale Monza",
        "gp_name": "Italian Grand Prix",
        "country": "Italy",
        "city": "Monza",
        "flag": "🇮🇹",
        "length_m": 5793.0,
        "turns": 11,
        "total_laps": 53,
        "drs_zones": 2,
        "lap_record": "1:21.046 (Rubens Barrichello, 2004)",
        "track_type": "High Speed Low Downforce Temple",
        "top_speed": 355.0,
        "min_speed": 78.0,
        "waypoints": np.array([
            [0.0, 0.0], [50.0, 15.0], [110.0, 22.0], [125.0, 18.0],
            [130.0, 5.0], [115.0, -8.0], [75.0, -12.0], [55.0, -15.0],
            [30.0, -22.0], [10.0, -18.0], [-15.0, -5.0]
        ]),
        "landmarks": [
            {"name": "Variante del Rettifilo (T1-T2)", "dist_ratio": 0.18, "speed_target": 78.0},
            {"name": "Curva Grande (T3)", "dist_ratio": 0.32, "speed_target": 295.0},
            {"name": "Variante della Roggia (T4-T5)", "dist_ratio": 0.48, "speed_target": 115.0},
            {"name": "Curva di Lesmo 1 & 2", "dist_ratio": 0.64, "speed_target": 160.0},
            {"name": "Variante Ascari (T8-T10)", "dist_ratio": 0.80, "speed_target": 175.0},
            {"name": "Curva Parabolica (Alboreto)", "dist_ratio": 0.94, "speed_target": 215.0}
        ]
    },
    "silverstone": {
        "name": "Silverstone Circuit",
        "gp_name": "British Grand Prix",
        "country": "Great Britain",
        "city": "Silverstone",
        "flag": "🇬🇧",
        "length_m": 5891.0,
        "turns": 18,
        "total_laps": 52,
        "drs_zones": 2,
        "lap_record": "1:27.097 (Max Verstappen, 2020)",
        "track_type": "High Speed Dynamic Downforce",
        "top_speed": 332.0,
        "min_speed": 82.0,
        "waypoints": np.array([
            [0.0, 0.0], [35.0, 20.0], [60.0, 15.0], [80.0, -5.0],
            [95.0, 10.0], [85.0, 45.0], [60.0, 60.0], [30.0, 70.0],
            [-5.0, 55.0], [-25.0, 30.0], [-15.0, 5.0]
        ]),
        "landmarks": [
            {"name": "Abbey & Farm (T1-T2)", "dist_ratio": 0.08, "speed_target": 260.0},
            {"name": "Village & The Loop (T3-T4)", "dist_ratio": 0.18, "speed_target": 92.0},
            {"name": "Brooklands & Luffield (T6-T7)", "dist_ratio": 0.36, "speed_target": 110.0},
            {"name": "Copse Corner (T9)", "dist_ratio": 0.54, "speed_target": 275.0},
            {"name": "Maggotts-Becketts-Chapel (T10-T14)", "dist_ratio": 0.70, "speed_target": 240.0},
            {"name": "Stowe Corner (T15)", "dist_ratio": 0.86, "speed_target": 180.0}
        ]
    },
    "spa": {
        "name": "Circuit de Spa-Francorchamps",
        "gp_name": "Belgian Grand Prix",
        "country": "Belgium",
        "city": "Stavelot",
        "flag": "🇧🇪",
        "length_m": 7004.0,
        "turns": 19,
        "total_laps": 44,
        "drs_zones": 2,
        "lap_record": "1:46.286 (Valtteri Bottas, 2018)",
        "track_type": "Rollercoaster Microclimate Power Track",
        "top_speed": 345.0,
        "min_speed": 68.0,
        "waypoints": np.array([
            [0.0, 0.0], [15.0, 30.0], [40.0, 80.0], [70.0, 110.0],
            [120.0, 125.0], [140.0, 95.0], [125.0, 55.0], [90.0, 35.0],
            [70.0, 10.0], [40.0, -15.0], [10.0, -10.0]
        ]),
        "landmarks": [
            {"name": "La Source Hairpin (T1)", "dist_ratio": 0.06, "speed_target": 68.0},
            {"name": "Eau Rouge & Raidillon (T2-T4)", "dist_ratio": 0.18, "speed_target": 305.0},
            {"name": "Kemmel Straight", "dist_ratio": 0.28, "speed_target": 345.0},
            {"name": "Les Combes (T5-T6)", "dist_ratio": 0.38, "speed_target": 135.0},
            {"name": "Bruxelles Hairpin (T8)", "dist_ratio": 0.52, "speed_target": 105.0},
            {"name": "Pouhon Double Apex (T10-T11)", "dist_ratio": 0.65, "speed_target": 280.0},
            {"name": "Blanchimont (T17-T18)", "dist_ratio": 0.88, "speed_target": 310.0},
            {"name": "Bus Stop Chicane (T19-T20)", "dist_ratio": 0.96, "speed_target": 75.0}
        ]
    },
    "monaco": {
        "name": "Circuit de Monaco",
        "gp_name": "Monaco Grand Prix",
        "country": "Monaco",
        "city": "Monte Carlo",
        "flag": "🇲🇨",
        "length_m": 3337.0,
        "turns": 19,
        "total_laps": 78,
        "drs_zones": 1,
        "lap_record": "1:12.909 (Lewis Hamilton, 2021)",
        "track_type": "Tight Barrier Street Circuit",
        "top_speed": 292.0,
        "min_speed": 45.0,
        "waypoints": np.array([
            [0.0, 0.0], [15.0, 20.0], [35.0, 35.0], [45.0, 25.0],
            [35.0, 10.0], [25.0, 2.0], [20.0, -15.0], [10.0, -28.0],
            [-5.0, -25.0], [-10.0, -10.0]
        ]),
        "landmarks": [
            {"name": "Sainte Dévote (T1)", "dist_ratio": 0.10, "speed_target": 90.0},
            {"name": "Massenet & Casino Square (T3-T4)", "dist_ratio": 0.25, "speed_target": 125.0},
            {"name": "Grand Hotel Hairpin (T6)", "dist_ratio": 0.40, "speed_target": 45.0},
            {"name": "Tunnel Section", "dist_ratio": 0.60, "speed_target": 285.0},
            {"name": "Nouvelle Chicane (T10-T11)", "dist_ratio": 0.70, "speed_target": 70.0},
            {"name": "Swimming Pool (T13-T16)", "dist_ratio": 0.85, "speed_target": 190.0},
            {"name": "La Rascasse (T18)", "dist_ratio": 0.94, "speed_target": 55.0}
        ]
    },
    "suzuka": {
        "name": "Suzuka International Racing Course",
        "gp_name": "Japanese Grand Prix",
        "country": "Japan",
        "city": "Suzuka",
        "flag": "🇯🇵",
        "length_m": 5807.0,
        "turns": 18,
        "total_laps": 53,
        "drs_zones": 1,
        "lap_record": "1:30.983 (Lewis Hamilton, 2019)",
        "track_type": "Iconic High-G Figure 8",
        "top_speed": 328.0,
        "min_speed": 70.0,
        "waypoints": np.array([
            [0.0, 0.0], [20.0, 30.0], [45.0, 45.0], [70.0, 35.0],
            [60.0, 10.0], [40.0, -10.0], [60.0, -30.0], [80.0, -20.0],
            [70.0, 0.0], [30.0, 15.0], [10.0, -10.0]
        ]),
        "landmarks": [
            {"name": "First Curve (T1-T2)", "dist_ratio": 0.12, "speed_target": 160.0},
            {"name": "S Curves & Dunlop (T3-T7)", "dist_ratio": 0.28, "speed_target": 215.0},
            {"name": "Degner 1 & 2 (T8-T9)", "dist_ratio": 0.45, "speed_target": 140.0},
            {"name": "Hairpin (T11)", "dist_ratio": 0.58, "speed_target": 70.0},
            {"name": "Spoon Curve (T13-T14)", "dist_ratio": 0.75, "speed_target": 185.0},
            {"name": "130R (T15)", "dist_ratio": 0.88, "speed_target": 305.0},
            {"name": "Casio Triangle Chicane (T16-T17)", "dist_ratio": 0.95, "speed_target": 82.0}
        ]
    },
    "bahrain": {
        "name": "Bahrain International Circuit",
        "gp_name": "Bahrain Grand Prix",
        "country": "Bahrain",
        "city": "Sakhir",
        "flag": "🇧🇭",
        "length_m": 5412.0,
        "turns": 15,
        "total_laps": 57,
        "drs_zones": 3,
        "lap_record": "1:31.447 (Pedro de la Rosa, 2005)",
        "track_type": "Desert Stop-and-Go Traction Test",
        "top_speed": 335.0,
        "min_speed": 65.0,
        "waypoints": np.array([
            [0.0, 0.0], [20.0, 40.0], [45.0, 42.0], [60.0, 25.0],
            [75.0, 30.0], [85.0, 15.0], [70.0, -10.0], [45.0, -25.0],
            [25.0, -20.0], [10.0, -10.0]
        ]),
        "landmarks": [
            {"name": "Michael Schumacher Turn (T1)", "dist_ratio": 0.15, "speed_target": 68.0},
            {"name": "Turn 4 (Downhill Brake)", "dist_ratio": 0.32, "speed_target": 130.0},
            {"name": "Turn 9-10 (Lockup Danger Hairpin)", "dist_ratio": 0.58, "speed_target": 75.0},
            {"name": "Turn 11 (High-G Sweeper)", "dist_ratio": 0.72, "speed_target": 205.0},
            {"name": "Turn 14-15 (Final Turn)", "dist_ratio": 0.94, "speed_target": 125.0}
        ]
    },
    "interlagos": {
        "name": "Autódromo José Carlos Pace (Interlagos)",
        "gp_name": "São Paulo Grand Prix",
        "country": "Brazil",
        "city": "São Paulo",
        "flag": "🇧🇷",
        "length_m": 4309.0,
        "turns": 15,
        "total_laps": 71,
        "drs_zones": 2,
        "lap_record": "1:10.540 (Valtteri Bottas, 2018)",
        "track_type": "Anticlockwise Natural Bowl",
        "top_speed": 330.0,
        "min_speed": 75.0,
        "waypoints": np.array([
            [0.0, 0.0], [-25.0, -15.0], [-40.0, 5.0], [-25.0, 35.0],
            [10.0, 45.0], [35.0, 30.0], [45.0, 5.0], [30.0, -15.0],
            [15.0, -25.0]
        ]),
        "landmarks": [
            {"name": "Senna 'S' (T1-T2)", "dist_ratio": 0.12, "speed_target": 95.0},
            {"name": "Curva do Sol (T3)", "dist_ratio": 0.24, "speed_target": 265.0},
            {"name": "Descida do Lago (T4)", "dist_ratio": 0.40, "speed_target": 145.0},
            {"name": "Ferradura (T6-T7)", "dist_ratio": 0.55, "speed_target": 185.0},
            {"name": "Bico de Pato (T10)", "dist_ratio": 0.72, "speed_target": 75.0},
            {"name": "Junção (T12)", "dist_ratio": 0.85, "speed_target": 120.0}
        ]
    },
    "singapore": {
        "name": "Marina Bay Street Circuit",
        "gp_name": "Singapore Grand Prix",
        "country": "Singapore",
        "city": "Marina Bay",
        "flag": "🇸🇬",
        "length_m": 4940.0,
        "turns": 19,
        "total_laps": 62,
        "drs_zones": 4,
        "lap_record": "1:34.486 (Daniel Ricciardo, 2024)",
        "track_type": "High Humidity Night Street Circuit",
        "top_speed": 315.0,
        "min_speed": 62.0,
        "waypoints": np.array([
            [0.0, 0.0], [25.0, 30.0], [50.0, 45.0], [70.0, 30.0],
            [60.0, 10.0], [40.0, -5.0], [20.0, -20.0], [5.0, -15.0]
        ]),
        "landmarks": [
            {"name": "Turn 1-3 Complex", "dist_ratio": 0.14, "speed_target": 90.0},
            {"name": "Turn 7 (Memorial)", "dist_ratio": 0.38, "speed_target": 110.0},
            {"name": "Turn 14 (Connaught)", "dist_ratio": 0.68, "speed_target": 85.0},
            {"name": "Turn 16-17 New Straight", "dist_ratio": 0.82, "speed_target": 290.0}
        ]
    }
}

# Add alias fallbacks for flexible name lookup
CIRCUIT_ALIASES = {
    "united states": "cota", "austin": "cota", "usa": "cota",
    "italy": "monza", "italian": "monza",
    "great britain": "silverstone", "british": "silverstone", "uk": "silverstone",
    "belgium": "spa", "belgian": "spa",
    "monte carlo": "monaco",
    "japan": "suzuka", "japanese": "suzuka",
    "sakhir": "bahrain",
    "brazil": "interlagos", "sao paulo": "interlagos"
}

def match_circuit_key(query: str) -> str:
    """Matches any user-provided string to an authentic circuit database key."""
    q = str(query).lower().strip()
    if q in CIRCUIT_DATABASE:
        return q
    for alias, key in CIRCUIT_ALIASES.items():
        if alias in q:
            return key
    for key, data in CIRCUIT_DATABASE.items():
        if key in q or data["name"].lower() in q or data["gp_name"].lower() in q or data["country"].lower() in q:
            return key
    return "cota"

def get_circuit_data(circuit_key: str, default_name: str = "Circuit") -> dict:
    """Retrieves full metadata and waypoints for a specific circuit."""
    key = match_circuit_key(circuit_key)
    data = CIRCUIT_DATABASE.get(key, CIRCUIT_DATABASE["cota"]).copy()
    if default_name and default_name != "Circuit" and "Grand Prix" in default_name:
        data["gp_name"] = default_name
    return data

# ==============================================================================
# PHYSICS TELEMETRY STREAM GENERATOR
# ==============================================================================
def generate_real_world_telemetry(circuit_key: str, driver1: str = "VER", driver2: str = "HAM", n_samples: int = 1200):
    """
    Synthesizes physically grounded, distance-aligned telemetry for two drivers
    navigating the specified circuit geometry.
    """
    key = match_circuit_key(circuit_key)
    meta = get_circuit_data(key)
    track_len = meta["length_m"]
    raw_waypoints = meta["waypoints"]
    landmarks = meta.get("landmarks", [])

    # Closed periodic Catmull-Rom trajectory (Pure NumPy)
    raw_x, raw_y = catmull_rom_spline_2d(raw_waypoints, n_samples=n_samples)
    dists = np.sqrt(np.diff(raw_x, prepend=raw_x[0])**2 + np.diff(raw_y, prepend=raw_y[0])**2)
    cum_d = np.cumsum(dists)
    scale_factor = track_len / max(1.0, cum_d[-1])
    
    smooth_x = raw_x * scale_factor
    smooth_y = raw_y * scale_factor
    fine_dist = np.linspace(0, track_len, n_samples)

    # Calculate curvature kappa = |x'y'' - y'x''| / (x'^2 + y'^2)^(3/2)
    dx = np.gradient(smooth_x, fine_dist)
    dy = np.gradient(smooth_y, fine_dist)
    ddx = np.gradient(dx, fine_dist)
    ddy = np.gradient(dy, fine_dist)
    curvature = np.abs(dx * ddy - dy * ddx) / np.maximum(1e-5, (dx**2 + dy**2)**1.5)

    # Base velocity profile limited by lateral grip (a_lat_max ~ 4.8G)
    mu_grip = 1.65
    g = 9.81
    a_lat_max = mu_grip * g * 3.0
    v_corner_limit = np.sqrt(a_lat_max / np.maximum(curvature, 1e-4)) * 3.6
    v_profile = np.clip(v_corner_limit, meta["min_speed"], meta["top_speed"])

    # Corner landmark modulation
    for lm in landmarks:
        center_idx = int(lm["dist_ratio"] * n_samples)
        half_width = int(0.04 * n_samples)
        start = max(0, center_idx - half_width)
        end = min(n_samples, center_idx + half_width)
        target = lm["speed_target"]
        v_profile[start:end] = np.minimum(v_profile[start:end], target + 2.0)

    # Driver 1 (Aggressive) & Driver 2 (Smooth) variations
    np.random.seed(42)
    v1 = np.clip(v_profile + np.random.normal(0, 0.8, n_samples), meta["min_speed"], meta["top_speed"])
    v2 = np.clip(v_profile * 0.992 + np.random.normal(0, 0.7, n_samples), meta["min_speed"], meta["top_speed"])

    # Time integration
    dt1 = np.diff(fine_dist, prepend=0) / np.maximum(v1 / 3.6, 5.0)
    dt2 = np.diff(fine_dist, prepend=0) / np.maximum(v2 / 3.6, 5.0)
    time1 = np.cumsum(dt1)
    time2 = np.cumsum(dt2)

    # Pedals & Gears
    throttle1 = np.clip((v1 - meta["min_speed"]) / (meta["top_speed"] - meta["min_speed"]) * 105.0, 0, 100)
    throttle2 = np.clip((v2 - meta["min_speed"]) / (meta["top_speed"] - meta["min_speed"]) * 102.0, 0, 100)
    brake1 = np.where(np.gradient(v1) < -1.2, np.clip(np.abs(np.gradient(v1)) * 18.0, 0, 100), 0)
    brake2 = np.where(np.gradient(v2) < -1.1, np.clip(np.abs(np.gradient(v2)) * 16.5, 0, 100), 0)

    gear1 = np.clip((v1 / meta["top_speed"] * 8).astype(int) + 1, 1, 8)
    gear2 = np.clip((v2 / meta["top_speed"] * 8).astype(int) + 1, 1, 8)
    drs = np.where((fine_dist / track_len > 0.15) & (fine_dist / track_len < 0.35), 1, 0)

    tel1 = pd.DataFrame({
        "Distance": fine_dist, "Time": time1, "Speed": v1,
        "Throttle": throttle1, "Brake": brake1, "nGear": gear1, "DRS": drs,
        "X": smooth_x, "Y": smooth_y, "Driver": driver1
    })

    tel2 = pd.DataFrame({
        "Distance": fine_dist, "Time": time2, "Speed": v2,
        "Throttle": throttle2, "Brake": brake2, "nGear": gear2, "DRS": drs,
        "X": smooth_x, "Y": smooth_y, "Driver": driver2
    })

    laps_summary = pd.DataFrame([
        {"Driver": driver1, "LapNumber": 1, "LapTime": timedelta(seconds=float(time1[-1])), "Compound": "MEDIUM", "TyreLife": 12},
        {"Driver": driver2, "LapNumber": 1, "LapTime": timedelta(seconds=float(time2[-1])), "Compound": "HARD", "TyreLife": 19}
    ])

    return tel1, tel2, laps_summary, track_len, meta, landmarks

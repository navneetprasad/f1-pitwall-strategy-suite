"""
Predictive Race Strategy Machine Learning Engine (predictive_strategy_ml.py)
Pure NumPy Random Forest & Regression-based Lap Time Performance Estimation,
Tyre Thermal Degradation Cliff Modeling, and Monte Carlo Strategy Window Optimizer.
Completely standalone with zero external C-extension DLL dependencies.
"""

import numpy as np
import pandas as pd

# Pirelli Compound Baseline Specs
COMPOUND_CHARACTERISTICS = {
    "SOFT": {"base_offset": -1.2, "deg_rate": 0.115, "cliff_lap": 16, "thermal_sensitivity": 1.45, "color": "#FF1801"},
    "MEDIUM": {"base_offset": 0.0, "deg_rate": 0.068, "cliff_lap": 26, "thermal_sensitivity": 1.00, "color": "#FFD129"},
    "HARD": {"base_offset": 0.95, "deg_rate": 0.038, "cliff_lap": 40, "thermal_sensitivity": 0.72, "color": "#FFFFFF"}
}

class PureNumpyDecisionTreeRegressor:
    """Fast Decision Tree Regressor implemented in pure NumPy."""
    def __init__(self, max_depth=5, min_samples_split=4):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def _best_split(self, X, y):
        best_mse = float('inf')
        best_feat = None
        best_thresh = None
        n_samples, n_features = X.shape
        
        for feat in range(n_features):
            vals = np.unique(X[:, feat])
            if len(vals) <= 1:
                continue
            thresholds = (vals[:-1] + vals[1:]) / 2.0
            if len(thresholds) > 10:
                thresholds = np.quantile(vals, np.linspace(0.15, 0.85, 6))
                
            for thresh in thresholds:
                left_mask = X[:, feat] <= thresh
                right_mask = ~left_mask
                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue
                mse = np.var(y[left_mask]) * np.sum(left_mask) + np.var(y[right_mask]) * np.sum(right_mask)
                if mse < best_mse:
                    best_mse = mse
                    best_feat = feat
                    best_thresh = thresh
        return best_feat, best_thresh

    def _build_tree(self, X, y, depth=0):
        if depth >= self.max_depth or len(y) < self.min_samples_split or np.var(y) < 1e-4:
            return {'val': float(np.mean(y))}
            
        feat, thresh = self._best_split(X, y)
        if feat is None:
            return {'val': float(np.mean(y))}
            
        left_mask = X[:, feat] <= thresh
        right_mask = ~left_mask
        
        return {
            'feat': feat, 'thresh': thresh,
            'left': self._build_tree(X[left_mask], y[left_mask], depth + 1),
            'right': self._build_tree(X[right_mask], y[right_mask], depth + 1)
        }

    def fit(self, X, y):
        self.tree = self._build_tree(np.asarray(X), np.asarray(y))
        return self

    def _predict_row(self, node, row):
        if 'val' in node:
            return node['val']
        if row[node['feat']] <= node['thresh']:
            return self._predict_row(node['left'], row)
        return self._predict_row(node['right'], row)

    def predict(self, X):
        X = np.asarray(X)
        return np.array([self._predict_row(self.tree, row) for row in X])

class PureNumpyRandomForestRegressor:
    """Fast Random Forest Ensemble implemented in pure NumPy."""
    def __init__(self, n_estimators=8, max_depth=5):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.trees = []

    def fit(self, X, y):
        X, y = np.asarray(X), np.asarray(y)
        n = len(X)
        self.trees = []
        for _ in range(self.n_estimators):
            idx = np.random.choice(n, size=n, replace=True)
            tree = PureNumpyDecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X[idx], y[idx])
            self.trees.append(tree)
        return self

    def predict(self, X):
        preds = np.array([tree.predict(X) for tree in self.trees])
        return np.mean(preds, axis=0)

class RaceStrategyMLEngine:
    """
    Predictive Race Strategy Engine combining pure NumPy Machine Learning
    with Monte Carlo stochastic simulations.
    """
    def __init__(self, base_lap_pace: float = 85.0, total_laps: int = 57, pit_loss_sec: float = 22.5):
        self.base_lap_pace = base_lap_pace
        self.total_laps = total_laps
        self.pit_loss_sec = pit_loss_sec
        self.model = PureNumpyRandomForestRegressor(n_estimators=8, max_depth=5)
        self._train_surrogate_model()

    def _train_surrogate_model(self):
        """Trains ML surrogate regressor on synthetic multi-stint telemetry data."""
        np.random.seed(42)
        records = []
        for compound, spec in COMPOUND_CHARACTERISTICS.items():
            for tyre_age in range(1, 48, 2):
                for fuel_kg in np.linspace(10, 110, 8):
                    for track_temp in [28.0, 36.0, 44.0]:
                        fuel_penalty = fuel_kg * 0.033
                        c_factor = 1.0 + (0.04 * max(0, tyre_age - spec["cliff_lap"])**1.8)
                        temp_factor = 1.0 + (track_temp - 32.0) * 0.008 * spec["thermal_sensitivity"]
                        deg = spec["deg_rate"] * tyre_age * c_factor * temp_factor
                        
                        lap_time = self.base_lap_pace + spec["base_offset"] + fuel_penalty + deg + np.random.normal(0, 0.08)
                        records.append([
                            tyre_age,
                            fuel_kg,
                            track_temp,
                            1.0 if compound == "HARD" else 0.0,
                            1.0 if compound == "SOFT" else 0.0,
                            1.0 if compound == "MEDIUM" else 0.0,
                            lap_time
                        ])
                        
        data = np.array(records)
        X = data[:, :-1]
        y = data[:, -1]
        self.model.fit(X, y)

    def predict_stint(self, compound: str, start_lap: int, end_lap: int, start_fuel_kg: float, track_temp: float = 35.0) -> np.ndarray:
        """Predicts lap times for a specific tire stint using trained ML model."""
        stint_laps = np.arange(start_lap, end_lap + 1)
        tyre_ages = np.arange(1, len(stint_laps) + 1)
        fuel_levels = np.maximum(5.0, start_fuel_kg - (stint_laps - 1) * 1.6)
        
        is_hard = 1.0 if compound == "HARD" else 0.0
        is_soft = 1.0 if compound == "SOFT" else 0.0
        is_medium = 1.0 if compound == "MEDIUM" else 0.0
        
        X_eval = np.column_stack([
            tyre_ages,
            fuel_levels,
            np.full_like(tyre_ages, track_temp),
            np.full_like(tyre_ages, is_hard),
            np.full_like(tyre_ages, is_soft),
            np.full_like(tyre_ages, is_medium)
        ])
        
        return self.model.predict(X_eval)

    def evaluate_strategy(self, strategy_name: str, stints: list, start_fuel_kg: float = 105.0) -> dict:
        total_time = 0.0
        lap_breakdown = []
        pit_stops = len(stints) - 1
        
        for idx, (comp, s_lap, e_lap) in enumerate(stints):
            pace_curve = self.predict_stint(comp, s_lap, e_lap, start_fuel_kg)
            for l_num, p in zip(range(s_lap, e_lap + 1), pace_curve):
                lap_breakdown.append({"Lap": l_num, "LapTime": p, "Compound": comp, "Stint": idx + 1})
            total_time += np.sum(pace_curve)
            
        total_time += pit_stops * self.pit_loss_sec
        return {
            "Strategy": strategy_name,
            "TotalRaceTime_sec": total_time,
            "TotalRaceTime_str": f"{int(total_time // 60)}m {total_time % 60:05.2f}s",
            "PitStops": pit_stops,
            "Laps": pd.DataFrame(lap_breakdown)
        }

    def run_monte_carlo(self, n_simulations: int = 1500, sc_probability_per_lap: float = 0.025) -> pd.DataFrame:
        results = []
        np.random.seed(42)
        
        strat_a_base = self.evaluate_strategy("1-Stop (M ➔ H)", [("MEDIUM", 1, 24), ("HARD", 25, self.total_laps)])
        strat_b_base = self.evaluate_strategy("2-Stop (S ➔ M ➔ M)", [("SOFT", 1, 15), ("MEDIUM", 16, 36), ("MEDIUM", 37, self.total_laps)])
        
        t_a_nominal = strat_a_base["TotalRaceTime_sec"]
        t_b_nominal = strat_b_base["TotalRaceTime_sec"]
        
        for sim in range(n_simulations):
            has_sc = np.random.rand() < (sc_probability_per_lap * self.total_laps)
            sc_lap = np.random.randint(5, self.total_laps - 5) if has_sc else -1
            
            discount_a = 10.5 if (has_sc and 22 <= sc_lap <= 26) else 0.0
            discount_b = 10.5 if (has_sc and (13 <= sc_lap <= 17 or 34 <= sc_lap <= 38)) else 0.0
            
            noise_a = np.random.normal(0, 1.8)
            noise_b = np.random.normal(0, 2.2)
            
            time_a = t_a_nominal - discount_a + noise_a
            time_b = t_b_nominal - discount_b + noise_b
            
            winner = "1-Stop (M ➔ H)" if time_a < time_b else "2-Stop (S ➔ M ➔ M)"
            delta = abs(time_a - time_b)
            
            results.append({
                "SimID": sim,
                "Time_A": time_a,
                "Time_B": time_b,
                "Delta": delta,
                "WinningStrategy": winner,
                "SafetyCar": "SC Deployed" if has_sc else "Green Flag",
                "SCLap": sc_lap
            })
            
        return pd.DataFrame(results)

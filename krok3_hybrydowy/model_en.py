"""
Step 3: hybrid model (physics + ML correction) - as in the bioreactor case study (Chapter 6, Appendix D.1).

Architecture: hybrid_prediction = physics_prediction + ML_correction(residuals)
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

np.random.seed(42)
n = 200
heater_power_kw = np.random.uniform(5, 20, n)
time_min = np.random.uniform(10, 120, n)
ambient_temp_c = np.random.uniform(15, 30, n)   # NEW variable - the physics model doesn't know it
noise = np.random.normal(0, 1.5, n)

# "TRUTH" (the real process): power saturation + time + HEAT LOSS to the surroundings
# (the cooler the surroundings relative to the 25C reference, the greater the loss -> lower temperature)
temperature_c = (
    20 + 15 * np.tanh(heater_power_kw / 8) + 0.15 * time_min
    - 0.3 * (25 - ambient_temp_c)   # this term is "invisible" to our physics
    + noise
)

data = pd.DataFrame({
    "heater_power_kw": heater_power_kw,
    "time_min": time_min,
    "ambient_temp_c": ambient_temp_c,
})
y = temperature_c

X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.2, random_state=42)

# ============================================================
# APPROACH A: PHYSICS ONLY (simplified mechanistic model, does NOT know ambient_temp)
# ============================================================
def physical_model(power, time):
    return 20 + 15 * np.tanh(power / 8) + 0.15 * time

pred_physics_test = physical_model(X_test["heater_power_kw"], X_test["time_min"])
rmse_physics = np.sqrt(mean_squared_error(y_test, pred_physics_test))
r2_physics = r2_score(y_test, pred_physics_test)

# ============================================================
# APPROACH B: ML ONLY - a "black box" trained on the raw signal, with all features
# ============================================================
scaler_b = StandardScaler()
Xtr_b = scaler_b.fit_transform(X_train)
Xte_b = scaler_b.transform(X_test)
model_ml_only = MLPRegressor(hidden_layer_sizes=(8,), activation="tanh", solver="lbfgs",
                              max_iter=5000, random_state=42).fit(Xtr_b, y_train)
pred_ml_only = model_ml_only.predict(Xte_b)
rmse_ml_only = np.sqrt(mean_squared_error(y_test, pred_ml_only))
r2_ml_only = r2_score(y_test, pred_ml_only)

# ============================================================
# APPROACH C: HYBRID = physics + ML on the residuals
# ============================================================
# Step 1: compute the physics predictions on the training set
pred_physics_train = physical_model(X_train["heater_power_kw"], X_train["time_min"])

# Step 2: compute the residuals = what the physics does NOT explain
residuals_train = y_train - pred_physics_train

# Step 3: train a SMALL ML model to predict the residuals (not the raw temperature!)
# We use ALL features, including ambient_temp - the ML gets a chance to discover the missing effect
scaler_c = StandardScaler()
Xtr_c = scaler_c.fit_transform(X_train)
Xte_c = scaler_c.transform(X_test)
model_correction = MLPRegressor(hidden_layer_sizes=(4,), activation="tanh", solver="lbfgs",
                              max_iter=5000, random_state=42).fit(Xtr_c, residuals_train)

# Step 4: combine - hybrid = physics + ML correction
correction_test = model_correction.predict(Xte_c)
pred_hybrid = pred_physics_test + correction_test
rmse_hybrid = np.sqrt(mean_squared_error(y_test, pred_hybrid))
r2_hybrid = r2_score(y_test, pred_hybrid)

# ============================================================
# COMPARISON (like the benchmark table in Appendix D.1 of the book "Process Modeling in Pharma: From Zero to a Validated Model")
# ============================================================
print("=== COMPARISON OF THREE APPROACHES (on the TEST set) ===\n")
print(f"{'Approach':<35} {'RMSE (C)':<12} {'R^2':<10}")
print(f"{'-'*57}")
print(f"{'A: physics only':<35} {rmse_physics:<12.3f} {r2_physics:<10.3f}")
print(f"{'B: ML only (black box)':<35} {rmse_ml_only:<12.3f} {r2_ml_only:<10.3f}")
print(f"{'C: hybrid (physics+ML correction)':<35} {rmse_hybrid:<12.3f} {r2_hybrid:<10.3f}")

print(f"\nHybrid vs physics alone: {(1 - rmse_hybrid/rmse_physics)*100:.1f}% lower RMSE")
print(f"Hybrid vs pure ML:       {(1 - rmse_hybrid/rmse_ml_only)*100:.1f}% lower RMSE")

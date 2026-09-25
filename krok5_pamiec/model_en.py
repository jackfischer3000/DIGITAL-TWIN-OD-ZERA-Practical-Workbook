"""
Step 5: why a process with memory (inertia) needs a model with memory.

We simulate a reactor with thermal inertia (a first-order system):
    T[t+1] = T[t] + dt * k * (T_target(power[t]) - T[t]) + noise

Key property: the same instantaneous power can correspond to different temperatures,
depending on HISTORY (how long the heater has been running at that level).
A model without memory (seeing only current power) has no way to tell these apart.
"""
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

np.random.seed(42)

# --- Simulation of a process with inertia (steps every 1 minute, 400 minutes) ---
n_steps = 400
dt = 1.0
k = 0.08   # the system's time constant - how fast T "catches up" to the target value

def t_target(power):
    return 20 + 15 * np.tanh(power / 8)   # same saturation physics as before

# Power changes in steps roughly every 40 minutes (typical for an operator changing a setpoint)
power_series = np.zeros(n_steps)
level = np.random.uniform(5, 20)
for t in range(n_steps):
    if t % 40 == 0:
        level = np.random.uniform(5, 20)
    power_series[t] = level

T = np.zeros(n_steps)
T[0] = 20.0
noise = np.random.normal(0, 0.3, n_steps)
for t in range(1, n_steps):
    T[t] = T[t - 1] + dt * k * (t_target(power_series[t - 1]) - T[t - 1]) + noise[t]

data = pd.DataFrame({"time_step": np.arange(n_steps), "power": power_series, "T_true": T})

# Split: first 300 steps training, last 100 test (typical for time series -
# NOT a random split, since that would falsify the evaluation: the model can't "see the future")
split = 300
train = data.iloc[:split]
test = data.iloc[split:]

# ============================================================
# MODEL A: WITHOUT MEMORY - sees only current power, predicts current T
# ============================================================
scaler_a = StandardScaler()
Xtr_a = scaler_a.fit_transform(train[["power"]])
Xte_a = scaler_a.transform(test[["power"]])
model_a = MLPRegressor(hidden_layer_sizes=(8,), activation="tanh", solver="lbfgs",
                        max_iter=5000, random_state=42).fit(Xtr_a, train["T_true"])
pred_a = model_a.predict(Xte_a)
rmse_a = np.sqrt(mean_squared_error(test["T_true"], pred_a))

# ============================================================
# MODEL B: WITH MEMORY - sees current power AND T from the previous step
# This is the core of what every recurrent network (RNN/GRU) does: the state from the previous
# step comes back as input to the next one. GRU adds "gates" on top of this, which
# learn how much of the previous state to keep and how much to overwrite - but the underlying
# mechanism is exactly what's shown here.
# ============================================================
data["T_previous"] = data["T_true"].shift(1)
data_b = data.dropna()
train_b = data_b.iloc[:split - 1]
test_b = data_b.iloc[split - 1:]

scaler_b = StandardScaler()
Xtr_b = scaler_b.fit_transform(train_b[["power", "T_previous"]])
Xte_b = scaler_b.transform(test_b[["power", "T_previous"]])
model_b = MLPRegressor(hidden_layer_sizes=(8,), activation="tanh", solver="lbfgs",
                        max_iter=5000, random_state=42).fit(Xtr_b, train_b["T_true"])

# IMPORTANT: in real "live" prediction you don't have the true future T_previous -
# the model has to use its OWN previous prediction (recurrent, step-by-step prediction).
pred_b = []
T_previous_simulated = test_b["T_previous"].iloc[0]
for _, row in test_b.iterrows():
    x_df = pd.DataFrame([[row["power"], T_previous_simulated]], columns=["power", "T_previous"])
    x = scaler_b.transform(x_df)
    p = model_b.predict(x)[0]
    pred_b.append(p)
    T_previous_simulated = p   # its own prediction becomes the "memory" for the next step
pred_b = np.array(pred_b)
rmse_b = np.sqrt(mean_squared_error(test_b["T_true"], pred_b))

print("=== COMPARISON: model without memory vs model with memory ===\n")
print(f"Model A (no memory, current power only):     RMSE = {rmse_a:.3f} C")
print(f"Model B (with memory, own T_previous):        RMSE = {rmse_b:.3f} C")
print(f"\nImprovement: {(1 - rmse_b/rmse_a)*100:.1f}% lower RMSE from adding memory")

# save to file for visualization
results = test.copy()
results["pred_A_no_memory"] = pred_a
results_b = test_b.copy()
results_b["pred_B_with_memory"] = pred_b
results.to_csv("krok5_pamiec/wyniki_A_en.csv", index=False)
results_b.to_csv("krok5_pamiec/wyniki_B_en.csv", index=False)

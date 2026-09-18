"""Visualization: how the ML correction removes the systematic error of the physics model."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

np.random.seed(42)
n = 200
heater_power_kw = np.random.uniform(5, 20, n)
time_min = np.random.uniform(10, 120, n)
ambient_temp_c = np.random.uniform(15, 30, n)
noise = np.random.normal(0, 1.5, n)
temperature_c = (20 + 15 * np.tanh(heater_power_kw / 8) + 0.15 * time_min
                  - 0.3 * (25 - ambient_temp_c) + noise)

data = pd.DataFrame({"heater_power_kw": heater_power_kw, "time_min": time_min,
                      "ambient_temp_c": ambient_temp_c})
y = temperature_c
X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.2, random_state=42)

def physical_model(power, time):
    return 20 + 15 * np.tanh(power / 8) + 0.15 * time

pred_physics_train = physical_model(X_train["heater_power_kw"], X_train["time_min"])
pred_physics_test = physical_model(X_test["heater_power_kw"], X_test["time_min"])
residuals_train = y_train - pred_physics_train
residuals_test_before = y_test - pred_physics_test

scaler = StandardScaler()
Xtr_s = scaler.fit_transform(X_train)
Xte_s = scaler.transform(X_test)
model_correction = MLPRegressor(hidden_layer_sizes=(4,), activation="tanh", solver="lbfgs",
                              max_iter=5000, random_state=42).fit(Xtr_s, residuals_train)
correction_test = model_correction.predict(Xte_s)
residuals_test_after = residuals_test_before - correction_test

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

ax = axes[0]
ax.scatter(X_test["ambient_temp_c"], residuals_test_before, alpha=0.7, color="#dc2626", edgecolor="white", s=60)
ax.axhline(0, color="black", linestyle="--", linewidth=1)
z = np.polyfit(X_test["ambient_temp_c"], residuals_test_before, 1)
xs = np.linspace(15, 30, 50)
ax.plot(xs, np.polyval(z, xs), color="#dc2626", linewidth=2, alpha=0.5, label="trend")
ax.set_xlabel("Ambient temperature (°C)")
ax.set_ylabel("Residual = true − predicted (°C)")
ax.set_title("BEFORE correction (physics only)\nclear trend = an effect the physics doesn't know about")
ax.legend()

ax = axes[1]
ax.scatter(X_test["ambient_temp_c"], residuals_test_after, alpha=0.7, color="#16a34a", edgecolor="white", s=60)
ax.axhline(0, color="black", linestyle="--", linewidth=1)
ax.set_xlabel("Ambient temperature (°C)")
ax.set_title("AFTER ML correction\nthe trend is gone — only noise remains")

plt.suptitle("Physics residuals as a function of ambient temperature — before and after ML correction", y=1.03)
plt.tight_layout()
plt.savefig("krok3_hybrydowy/korekta_rezyduow_en.png", dpi=130, bbox_inches="tight")
print("Saved: krok3_hybrydowy/korekta_rezyduow_en.png")

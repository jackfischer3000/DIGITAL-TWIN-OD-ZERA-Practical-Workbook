"""Visualization of RMSE and R^2 for the example in model_en.py."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

np.random.seed(42)
n = 200
heater_power_kw = np.random.uniform(5, 20, n)
time_min = np.random.uniform(10, 120, n)
noise = np.random.normal(0, 1.5, n)
temperature_c = 20 + 1.8 * heater_power_kw + 0.15 * time_min + noise

X = pd.DataFrame({"heater_power_kw": heater_power_kw, "time_min": time_min})
y = temperature_c

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression().fit(X_train, y_train)
y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
residuals = y_test - y_pred

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1: predicted vs true
ax = axes[0]
ax.scatter(y_test, y_pred, alpha=0.7, color="#2563eb", edgecolor="white", s=60)
lims = [min(y_test.min(), y_pred.min()) - 1, max(y_test.max(), y_pred.max()) + 1]
ax.plot(lims, lims, "--", color="gray", label="perfect prediction (y=x)")
ax.fill_between(lims, [l - rmse for l in lims], [l + rmse for l in lims],
                 color="#2563eb", alpha=0.1, label=f"±RMSE band ({rmse:.2f}°C)")
ax.set_xlabel("True temperature (°C)")
ax.set_ylabel("Predicted temperature (°C)")
ax.set_title(f"Prediction vs reality\nR² = {r2:.3f}")
ax.legend()
ax.set_xlim(lims); ax.set_ylim(lims)

# Plot 2: error distribution (residuals)
ax = axes[1]
ax.hist(residuals, bins=15, color="#2563eb", alpha=0.7, edgecolor="white")
ax.axvline(0, color="black", linestyle="--", linewidth=1)
ax.axvline(rmse, color="red", linestyle=":", label=f"+RMSE = {rmse:.2f}")
ax.axvline(-rmse, color="red", linestyle=":", label=f"-RMSE = {-rmse:.2f}")
ax.set_xlabel("Prediction error (°C) = true − predicted")
ax.set_ylabel("Number of observations")
ax.set_title("Error distribution (residuals)")
ax.legend()

plt.tight_layout()
plt.savefig("krok1_regresja/rmse_r2_wizualizacja_en.png", dpi=130)
print("Saved: krok1_regresja/rmse_r2_wizualizacja_en.png")
print(f"RMSE={rmse:.3f}, R2={r2:.3f}")

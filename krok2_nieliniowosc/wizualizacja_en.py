"""Visualization: where linear regression is systematically wrong, and the NN model isn't."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

np.random.seed(42)
n = 200
heater_power_kw = np.random.uniform(5, 20, n)
time_min = np.random.uniform(10, 120, n)
noise = np.random.normal(0, 1.5, n)
temperature_c = 20 + 15 * np.tanh(heater_power_kw / 8) + 0.15 * time_min + noise

X = pd.DataFrame({"heater_power_kw": heater_power_kw, "time_min": time_min})
y = temperature_c
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model_linear = LinearRegression().fit(X_train, y_train)
pred_linear = model_linear.predict(X_test)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)
model_nn = MLPRegressor(hidden_layer_sizes=(8,), activation="tanh", solver="lbfgs",
                         max_iter=5000, random_state=42).fit(X_train_s, y_train)
pred_nn = model_nn.predict(X_test_s)

resid_lin = y_test - pred_linear
resid_nn = y_test - pred_nn
power_test = X_test["heater_power_kw"].values

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

for ax, resid, title in [(axes[0], resid_lin, "Linear regression"), (axes[1], resid_nn, "Neural network (MLP)")]:
    ax.scatter(power_test, resid, alpha=0.7, color="#2563eb", edgecolor="white", s=60)
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Heater power (kW)")
    ax.set_title(title)
axes[0].set_ylabel("Error (residual) = true − predicted (°C)")

plt.suptitle("Prediction error as a function of heater power — looking for a PATTERN in the points, not a random cloud", y=1.02)
plt.tight_layout()
plt.savefig("krok2_nieliniowosc/residua_porownanie_en.png", dpi=130, bbox_inches="tight")
print("Saved: krok2_nieliniowosc/residua_porownanie_en.png")

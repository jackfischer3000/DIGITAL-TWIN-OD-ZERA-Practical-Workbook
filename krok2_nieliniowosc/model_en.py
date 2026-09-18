"""
Step 2: nonlinearity - where linear regression fails, and the first neural network.
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
noise = np.random.normal(0, 1.5, n)

# This time: SATURATION at high power (tanh) instead of a simple linear relationship.
# Physical interpretation: above a certain power, the temperature increase slows down (losses grow, cooling kicks in).
temperature_c = 20 + 15 * np.tanh(heater_power_kw / 8) + 0.15 * time_min + noise

X = pd.DataFrame({"heater_power_kw": heater_power_kw, "time_min": time_min})
y = temperature_c

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- MODEL A: linear regression (as in step 1) ---
model_linear = LinearRegression().fit(X_train, y_train)
pred_linear = model_linear.predict(X_test)
rmse_linear = np.sqrt(mean_squared_error(y_test, pred_linear))
r2_linear = r2_score(y_test, pred_linear)

print("=== MODEL A: linear regression ===")
print(f"RMSE: {rmse_linear:.3f} C")
print(f"R^2:  {r2_linear:.3f}")

# --- MODEL B: a small neural network (1 hidden layer, 8 neurons) ---
# MLP = Multi-Layer Perceptron. Each neuron computes: weighted sum of inputs -> activation function (here: relu/tanh).
# It's this activation function that introduces NONLINEARITY - without it, a network made
# entirely of linear layers would still just be linear regression, regardless of the number of layers.
#
# IMPORTANT: a neural network requires SCALED features (mean=0, std=1).
# Without this, given the different input ranges (power: 5-20, time: 10-120), gradient training
# is unstable and can "explode" (overflow). Linear regression doesn't have this problem.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # fit ONLY on training data - the test set can't "leak" into scaling
X_test_scaled = scaler.transform(X_test)

# solver="lbfgs" (not the default "adam"): adam/sgd are stochastic methods designed
# for large datasets (mini-batch training). For a small dataset (160 observations),
# lbfgs (a quasi-Newton, full-batch method) converges faster and more stably - this is
# scikit-learn's own official recommendation for small data.
model_nn = MLPRegressor(
    hidden_layer_sizes=(8,),   # one hidden layer, 8 neurons
    activation="tanh",
    solver="lbfgs",
    max_iter=5000,
    random_state=42,
)
model_nn.fit(X_train_scaled, y_train)
pred_nn = model_nn.predict(X_test_scaled)
rmse_nn = np.sqrt(mean_squared_error(y_test, pred_nn))
r2_nn = r2_score(y_test, pred_nn)

print("\n=== MODEL B: neural network (MLP, 1 layer, 8 neurons) ===")
print(f"RMSE: {rmse_nn:.3f} C")
print(f"R^2:  {r2_nn:.3f}")

print(f"\n=== COMPARISON ===")
print(f"Linear regression: RMSE={rmse_linear:.3f}, R^2={r2_linear:.3f}")
print(f"Neural network:    RMSE={rmse_nn:.3f}, R^2={r2_nn:.3f}")
print(f"RMSE improvement: {(1 - rmse_nn/rmse_linear)*100:.1f}%")

"""
Step 1: the simplest complete ML model — linear regression.
Goal: go through the full cycle (data -> training -> validation -> interpretation) once, on a simple example.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# --- 1. DATA ---
# In a real project, data comes from the historian/SCADA (see Chapter 5 of the book "Process Modeling in Pharma: From Zero to a Validated Model": "Data in a GxP Environment").
# Here: we generate it synthetically, so we control the "true" relationship and can check
# whether the model is able to recover it.
np.random.seed(42)
n = 200

heater_power_kw = np.random.uniform(5, 20, n)     # kW
time_min = np.random.uniform(10, 120, n)          # minutes

# "True" physics (simplified): temperature rises with power and time, plus measurement noise
noise = np.random.normal(0, 1.5, n)
temperature_c = 20 + 1.8 * heater_power_kw + 0.15 * time_min + noise

data = pd.DataFrame({
    "heater_power_kw": heater_power_kw,
    "time_min": time_min,
    "temperature_c": temperature_c,
})

print("First 5 rows of data:")
print(data.head())
print(f"\nNumber of observations: {len(data)}")

# --- 2. TRAIN/TEST SPLIT ---
# Never evaluate a model on data it trained on - this is a basic rule (the equivalent of "OQ" in the book).
X = data[["heater_power_kw", "time_min"]]
y = data["temperature_c"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nTraining set: {len(X_train)} observations, test set: {len(X_test)} observations")

# --- 3. TRAINING (the equivalent of Chapter 9 - "building and training the model") ---
model = LinearRegression()
model.fit(X_train, y_train)

print(f"\nModel coefficients:")
print(f"  heater_power_kw: {model.coef_[0]:.3f}  (true value: 1.8)")
print(f"  time_min:        {model.coef_[1]:.3f}  (true value: 0.15)")
print(f"  intercept:       {model.intercept_:.3f}  (true value: 20)")

# --- 4. VALIDATION (the equivalent of Chapter 10 - "model validation / OQ") ---
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"\n--- Results on the TEST set (data the model never saw) ---")
print(f"RMSE: {rmse:.3f} C")
print(f"R^2:  {r2:.3f}  (1.0 = perfect fit)")

# --- 5. INTERPRETATION ---
print("\n--- Example prediction ---")
example = X_test.iloc[[0]]
true_value = y_test.iloc[0]
prediction = model.predict(example)[0]
print(f"Input: power={example['heater_power_kw'].values[0]:.1f} kW, time={example['time_min'].values[0]:.1f} min")
print(f"True temperature: {true_value:.2f} C")
print(f"Model prediction: {prediction:.2f} C")

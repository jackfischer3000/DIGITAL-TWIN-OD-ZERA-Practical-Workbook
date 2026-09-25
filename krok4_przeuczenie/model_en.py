"""
Step 4: overfitting and cross-validation.

Classic example: we fit polynomials of increasing degree to a SMALL, noisy
dataset. Low degree = too simple (underfitting). High degree = the model
"memorizes" the noise instead of learning the true relationship (overfitting).
"""
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

np.random.seed(7)
n = 25   # DELIBERATELY little data - overfitting is more dramatic and easier to see
power = np.random.uniform(5, 20, n)
noise = np.random.normal(0, 1.2, n)
temperature = 20 + 15 * np.tanh(power / 8) + noise

X = power.reshape(-1, 1)
y = temperature

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=7)

degrees = [1, 2, 3, 5, 9, 15]

print(f"{'Polynomial degree':<20} {'RMSE (train)':<15} {'RMSE (test)':<15}")
print("-" * 50)

results = []
for degree in degrees:
    model = make_pipeline(
        PolynomialFeatures(degree=degree),
        StandardScaler(),
        LinearRegression(),
    )
    model.fit(X_train, y_train)
    rmse_train = np.sqrt(mean_squared_error(y_train, model.predict(X_train)))
    rmse_test = np.sqrt(mean_squared_error(y_test, model.predict(X_test)))
    results.append((degree, rmse_train, rmse_test))
    print(f"{degree:<20} {rmse_train:<15.3f} {rmse_test:<15.3f}")

print("\nObservation: TRAINING RMSE drops almost to zero at high degree -")
print("the model perfectly 'memorizes' the 17 training points. TEST RMSE rises -")
print("the model doesn't generalize to new data. This is overfitting.")

# ============================================================
# PROBLEM with a single train/test split: for 25 points, the test result
# depends heavily on WHICH points ended up in the test set (split randomness).
# Let's check this - we only change the split's random_state.
# ============================================================
print("\n=== How unstable is the result of a single train/test split? ===")
print("(same model, degree=3, different random splits)\n")
for rs in [1, 2, 3, 4, 5]:
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=rs)
    model = make_pipeline(PolynomialFeatures(degree=3), StandardScaler(), LinearRegression())
    model.fit(Xtr, ytr)
    rmse = np.sqrt(mean_squared_error(yte, model.predict(Xte)))
    print(f"random_state={rs}: test RMSE = {rmse:.3f}")

# ============================================================
# CROSS-VALIDATION (k-fold cross-validation): instead of ONE split,
# we divide the data into k parts, each in turn serving as the test set, the rest as training.
# We average the k results -> a more stable, more reliable estimate.
# ============================================================
print("\n=== Cross-validation (5-fold) for each polynomial degree ===\n")
print(f"{'Degree':<10} {'CV RMSE (mean)':<20} {'CV RMSE (std. dev.)':<20}")
print("-" * 50)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_results = []
for degree in degrees:
    model = make_pipeline(PolynomialFeatures(degree=degree), StandardScaler(), LinearRegression())
    # scikit-learn computes scoring as "neg_mean_squared_error" (higher=better), hence the minus sign
    scores = cross_val_score(model, X, y, cv=kf, scoring="neg_root_mean_squared_error")
    cv_rmse = -scores
    cv_results.append((degree, cv_rmse.mean(), cv_rmse.std()))
    print(f"{degree:<10} {cv_rmse.mean():<20.3f} {cv_rmse.std():<20.3f}")

best = min(cv_results, key=lambda w: w[1])
print(f"\nBest degree per CV: {best[0]} (CV RMSE = {best[1]:.3f})")
print("It's this number - not the result on a single test - that should decide model complexity.")

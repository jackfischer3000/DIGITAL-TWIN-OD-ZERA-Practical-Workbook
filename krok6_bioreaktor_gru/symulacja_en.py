"""
Step 6: bioreactor simulation - a replica of case study D.1 from the book "Process Modeling in Pharma: From Zero to a Validated Model" (DO prediction).

Physics: dissolved oxygen mass balance
    dDO/dt = OTR - OUR
    OTR (oxygen transfer rate) = kLa * (DO_sat - DO)          [Van't Riet-style kLa correlation]
    OUR (oxygen uptake rate)   = qO2 * X * DO/(DO+K_do)        [oxygen consumption by biomass]

"Hidden" effect (which the simple physical model does NOT know about): metabolic overflow mode -
when the cumulative combination of biomass density and feeding history crosses a threshold,
the cells increase their oxygen consumption (qO2 rises). This is exactly the mechanism described
in the book's Introduction: "a combination of inoculum density and feeding history" -> a delayed
DO drop. Detecting this requires MEMORY (hence GRU), not just current values.
"""
import numpy as np
import pandas as pd

DT = 1.0          # min
DO_SAT = 100.0    # % saturation
XMAX = 40.0       # g/L, carrying capacity
MU_MAX = 0.018    # 1/min, biomass growth rate
QO2_BASE = 0.04   # baseline oxygen consumption per gram of biomass
K_DO = 5.0        # Monod constant for oxygen limitation
C_KLA = 0.08      # kLa correlation coefficient (Van't Riet style: kLa ~ power^a * flow^b)


def kla(stirring_rpm, aeration_vvm):
    return C_KLA * (stirring_rpm / 500.0) ** 2.2 * (aeration_vvm / 1.0) ** 0.5


def step_series(n_steps, levels, every):
    """Generates a step series (an operator periodically changes the setpoint)."""
    series = np.zeros(n_steps)
    level = np.random.uniform(*levels)
    for t in range(n_steps):
        if t % every == 0:
            level = np.random.uniform(*levels)
        series[t] = level
    return series


def simulate_batch(n_steps=480, seed=0):
    rng = np.random.default_rng(seed)
    np.random.seed(seed)

    stirring = step_series(n_steps, (250, 750), every=60)
    aeration = step_series(n_steps, (0.5, 2.0), every=60)
    feed_rate = step_series(n_steps, (0.0, 1.0), every=45)

    X = np.zeros(n_steps)
    X[0] = np.random.uniform(0.5, 3.0)   # inoculum density - different for each batch
    DO = np.zeros(n_steps)
    DO[0] = 90.0

    metabolic_stress = np.zeros(n_steps)
    metabolic_mode = np.zeros(n_steps)
    threshold = 55.0

    noise = rng.normal(0, 0.15, n_steps)

    for t in range(1, n_steps):
        # biomass growth (logistic)
        dX = MU_MAX * X[t - 1] * (1 - X[t - 1] / XMAX) * DT
        X[t] = X[t - 1] + dX

        # accumulation of "metabolic stress" - a combination of biomass and feeding
        metabolic_stress[t] = metabolic_stress[t - 1] * 0.985 + X[t] * feed_rate[t] * 0.06
        metabolic_mode[t] = 1.0 if metabolic_stress[t] > threshold else 0.0

        qo2_eff = QO2_BASE * (1.7 if metabolic_mode[t] else 1.0)

        otr = kla(stirring[t - 1], aeration[t - 1]) * (DO_SAT - DO[t - 1])
        our = qo2_eff * X[t] * DO[t - 1] / (DO[t - 1] + K_DO)
        DO[t] = np.clip(DO[t - 1] + DT * (otr - our) + noise[t], 0.0, 100.0)

    return pd.DataFrame({
        "t": np.arange(n_steps),
        "stirring_rpm": stirring,
        "aeration_vvm": aeration,
        "feed_rate": feed_rate,
        "X_biomass": X,
        "DO": DO,
        "metabolic_mode": metabolic_mode,   # for visualization/diagnostics only - NOT a model input
    })


if __name__ == "__main__":
    # quick calibration check
    df = simulate_batch(seed=1)
    print(df.describe())
    print(f"\nDO min={df['DO'].min():.1f}, max={df['DO'].max():.1f}")
    print(f"Fraction of time in metabolic mode: {df['metabolic_mode'].mean()*100:.1f}%")

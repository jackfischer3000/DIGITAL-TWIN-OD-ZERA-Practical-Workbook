"""Visualization: trajectory over time - truth vs model without memory vs model with memory."""
import pandas as pd
import matplotlib.pyplot as plt

results_a = pd.read_csv("krok5_pamiec/wyniki_A_en.csv")
results_b = pd.read_csv("krok5_pamiec/wyniki_B_en.csv")

fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

ax = axes[0]
ax.plot(results_a["time_step"], results_a["T_true"], color="#1e293b", linewidth=2, label="True T")
ax.plot(results_a["time_step"], results_a["pred_A_no_memory"], color="#dc2626", linewidth=1.5,
        linestyle="--", label="Model A: no memory (current power only)")
ax2 = ax.twinx()
ax2.plot(results_a["time_step"], results_a["power"], color="#94a3b8", alpha=0.5, linewidth=1, label="power (right axis)")
ax2.set_ylabel("Heater power (kW)", color="#94a3b8")
ax.set_ylabel("Temperature (°C)")
ax.set_title("Model WITHOUT memory — misses the truth at every power change (transient)")
ax.legend(loc="upper left", fontsize=8)

ax = axes[1]
ax.plot(results_b["time_step"], results_b["T_true"], color="#1e293b", linewidth=2, label="True T")
ax.plot(results_b["time_step"], results_b["pred_B_with_memory"], color="#16a34a", linewidth=1.5,
        linestyle="--", label="Model B: with memory (power + own previous T)")
ax.set_xlabel("Time step (minutes)")
ax.set_ylabel("Temperature (°C)")
ax.set_title("Model WITH memory — tracks the dynamics, not just the instantaneous value")
ax.legend(loc="upper left", fontsize=8)

plt.tight_layout()
plt.savefig("krok5_pamiec/pamiec_porownanie_en.png", dpi=130, bbox_inches="tight")
print("Saved: krok5_pamiec/pamiec_porownanie_en.png")

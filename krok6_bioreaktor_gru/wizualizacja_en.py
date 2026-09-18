"""Visualization: true DO vs predictions (physics / GRU / hybrid) at a 30-minute horizon."""
import numpy as np
import matplotlib.pyplot as plt
import torch

from symulacja_en import simulate_batch
from model_en import (
    normalize, physics_rollout, GRUPredictor, W, H,
    ds_train_bb, ds_train_res, destandardize,
)

model_bb = GRUPredictor()
model_bb.load_state_dict(torch.load("krok6_bioreaktor_gru/model_bb_en.pt"))
model_bb.eval()

model_res = GRUPredictor()
model_res.load_state_dict(torch.load("krok6_bioreaktor_gru/model_res_en.pt"))
model_res.eval()

# test batch with a clear metabolic episode (seed outside training, see model_en.py: test = seeds 12-15)
df = simulate_batch(seed=13)
df_norm = normalize(df)

times, pred_physics, pred_bb, pred_hybrid = [], [], [], []
with torch.no_grad():
    for t0 in range(W - 1, len(df) - H - 1):
        window = torch.from_numpy(df_norm.iloc[t0 - W + 1: t0 + 1].values.astype(np.float32)).unsqueeze(0)
        pf = physics_rollout(df, t0)
        pbb = destandardize(model_bb(window).item(), ds_train_bb)
        pres = destandardize(model_res(window).item(), ds_train_res)
        times.append(t0 + H)
        pred_physics.append(pf)
        pred_bb.append(pbb)
        pred_hybrid.append(pf + pres)

fig, ax = plt.subplots(figsize=(13, 6))
ax.plot(df["t"], df["DO"], color="#1e293b", linewidth=2.2, label="True DO", zorder=5)
ax.plot(times, pred_physics, color="#f59e0b", linewidth=1.5, linestyle="--", label="A: physics only")
ax.plot(times, pred_bb, color="#dc2626", linewidth=1.3, linestyle=":", label="B: GRU only (black box)")
ax.plot(times, pred_hybrid, color="#16a34a", linewidth=1.8, label="C: hybrid (physics + GRU)")

# shade the metabolic-mode periods
in_mode = df["metabolic_mode"].values
start = None
for t in range(len(in_mode)):
    if in_mode[t] and start is None:
        start = t
    if (not in_mode[t] or t == len(in_mode) - 1) and start is not None:
        ax.axvspan(start, t, color="#fca5a5", alpha=0.2)
        start = None

ax.set_xlabel("Time (minutes)")
ax.set_ylabel("DO (% saturation)")
ax.set_title("DO prediction at a 30-min horizon — test batch (red bands = metabolic mode, invisible to physics)")
ax.legend(loc="lower left", fontsize=9)

plt.tight_layout()
plt.savefig("krok6_bioreaktor_gru/benchmark_trajektoria_en.png", dpi=130, bbox_inches="tight")
print("Saved: krok6_bioreaktor_gru/benchmark_trajektoria_en.png")

"""
Step 6: replica of case study D.1 - DO prediction in a bioreactor, hybrid architecture
(mass balance + GRU), PyTorch. Benchmark: physics vs GRU (black box) vs hybrid,
at a 30-minute horizon - as in Chapters 10/11 and Appendix D.1 of the book "Process Modeling in Pharma: From Zero to a Validated Model".
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from symulacja_en import simulate_batch, kla, QO2_BASE, K_DO, DO_SAT, DT

torch.manual_seed(0)

W = 60   # history window (minutes) fed into the GRU
H = 30   # prediction horizon (minutes) - as in the book's benchmark

FEATURES = ["stirring_rpm", "aeration_vvm", "feed_rate", "X_biomass", "DO"]

# ============================================================
# 1. DATA: 16 batches (simulated), 12 training / 4 test
# ============================================================
batches = [simulate_batch(seed=s) for s in range(16)]
batches_train = batches[:12]
batches_test = batches[12:]

# normalization - computed ONLY on training data (as in steps 1-2, so as not to "leak" test data)
all_train = pd.concat(batches_train)
mean = all_train[FEATURES].mean()
std = all_train[FEATURES].std()


def normalize(df):
    return (df[FEATURES] - mean) / std


# DO (target) also needs to be standardized for the "raw" model - without this, training
# is unstable (the same reasons as in step 2: input/output scale for the network).
# Residuals are naturally smaller/closer to zero, so it's less critical for the hybrid model,
# but we do it consistently for both.
DO_MEAN = all_train["DO"].mean()
DO_STD = all_train["DO"].std()


# ============================================================
# 2. PHYSICAL MODEL (incomplete - does NOT know about the metabolic overflow mode)
#    Rolls out H steps ahead, assuming a known control schedule (stirring/aeration/feed)
#    - a standard assumption: the schedule is part of the process recipe, so it's known in advance.
# ============================================================
def physics_rollout(df, t0, horizon=H):
    do = df["DO"].iloc[t0]
    for t in range(t0 + 1, t0 + horizon + 1):
        x = df["X_biomass"].iloc[t]
        otr = kla(df["stirring_rpm"].iloc[t - 1], df["aeration_vvm"].iloc[t - 1]) * (DO_SAT - do)
        our = QO2_BASE * x * do / (do + K_DO)   # QO2_BASE constant - physics does NOT know about the metabolic mode
        do = np.clip(do + DT * (otr - our), 0.0, 100.0)
    return do


# ============================================================
# 3. DATASET FOR THE GRU: window [t0-W+1, t0] -> target at t0+H
# ============================================================
class WindowDataset(Dataset):
    def __init__(self, batches, mode):
        windows, raw_targets = [], []
        for df in batches:
            df_norm = normalize(df)
            physics_cache = {}
            for t0 in range(W - 1, len(df) - H - 1):
                window = df_norm.iloc[t0 - W + 1: t0 + 1].values.astype(np.float32)
                true_do = df["DO"].iloc[t0 + H]
                if mode == "residual":
                    if t0 not in physics_cache:
                        physics_cache[t0] = physics_rollout(df, t0)
                    target = true_do - physics_cache[t0]      # GRU learns the RESIDUAL
                else:
                    target = true_do                          # GRU learns the raw value (black box)
                windows.append(window)
                raw_targets.append(target)

        raw_targets = np.array(raw_targets, dtype=np.float32)
        # target standardization - same logic as in step 2: without it, network training is unstable
        self.target_mean = float(raw_targets.mean())
        self.target_std = float(raw_targets.std()) + 1e-6
        self.windows = windows
        self.targets = (raw_targets - self.target_mean) / self.target_std

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        return torch.from_numpy(self.windows[idx]), torch.tensor(self.targets[idx])


# ============================================================
# 4. GRU MODEL (PyTorch) - small, one layer, for regressing a single value
# ============================================================
class GRUPredictor(nn.Module):
    def __init__(self, n_features=len(FEATURES), hidden=24):
        super().__init__()
        self.gru = nn.GRU(input_size=n_features, hidden_size=hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        _, h_last = self.gru(x)         # h_last: hidden state after the whole sequence
        return self.head(h_last.squeeze(0)).squeeze(-1)


def train(model, dataset, epochs=15, batch_size=64, lr=1e-3):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    for epoch in range(epochs):
        model.train()
        loss_sum = 0.0
        for window, target in loader:
            opt.zero_grad()
            pred = model(window)
            loss = loss_fn(pred, target)
            loss.backward()
            opt.step()
            loss_sum += loss.item() * len(target)
        if (epoch + 1) % 5 == 0:
            print(f"  epoch {epoch+1}/{epochs}: training MSE = {loss_sum/len(dataset):.3f}")
    return model


print("=== Training model B: black-box GRU (raw DO value) ===")
ds_train_bb = WindowDataset(batches_train, mode="raw")
model_bb = train(GRUPredictor(), ds_train_bb)

print("\n=== Training model C: GRU on residuals (hybrid) ===")
ds_train_res = WindowDataset(batches_train, mode="residual")
model_res = train(GRUPredictor(), ds_train_res)


def destandardize(normalized_value, dataset):
    return normalized_value * dataset.target_std + dataset.target_mean


# ============================================================
# 5. EVALUATION on the TEST set (batches no model has seen)
# ============================================================
def evaluate(batches_test):
    errors_physics, errors_bb, errors_hybrid = [], [], []
    model_bb.eval()
    model_res.eval()
    with torch.no_grad():
        for df in batches_test:
            df_norm = normalize(df)
            for t0 in range(W - 1, len(df) - H - 1):
                window = torch.from_numpy(df_norm.iloc[t0 - W + 1: t0 + 1].values.astype(np.float32)).unsqueeze(0)
                true_value = df["DO"].iloc[t0 + H]

                pred_physics = physics_rollout(df, t0)
                pred_bb = destandardize(model_bb(window).item(), ds_train_bb)
                correction = destandardize(model_res(window).item(), ds_train_res)
                pred_hybrid = pred_physics + correction

                errors_physics.append(true_value - pred_physics)
                errors_bb.append(true_value - pred_bb)
                errors_hybrid.append(true_value - pred_hybrid)

    rmse = lambda b: float(np.sqrt(np.mean(np.square(b))))
    return rmse(errors_physics), rmse(errors_bb), rmse(errors_hybrid)


rmse_physics, rmse_bb, rmse_hybrid = evaluate(batches_test)

print(f"\n=== BENCHMARK (horizon {H} min, test set: {len(batches_test)} batches) ===\n")
print(f"{'Approach':<35} {'RMSE (%DO sat.)':<18}")
print("-" * 53)
print(f"{'A: physics only (no metabolic effect)':<35} {rmse_physics:<18.3f}")
print(f"{'B: GRU only (black box)':<35} {rmse_bb:<18.3f}")
print(f"{'C: hybrid (physics + GRU on residuals)':<35} {rmse_hybrid:<18.3f}")
print(f"\nHybrid vs physics: {(1-rmse_hybrid/rmse_physics)*100:.1f}% lower RMSE")
print(f"Hybrid vs pure GRU: {(1-rmse_hybrid/rmse_bb)*100:.1f}% lower RMSE")

torch.save(model_bb.state_dict(), "krok6_bioreaktor_gru/model_bb_en.pt")
torch.save(model_res.state_dict(), "krok6_bioreaktor_gru/model_res_en.pt")

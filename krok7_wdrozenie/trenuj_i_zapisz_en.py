"""
Step 7: we train the model (exactly as in step 6) and save ALL the artifacts
needed for serving: GRU weights, input scaling statistics, target normalization
statistics, the conformal prediction quantile, Mahalanobis statistics (for the OOD flag).

This is the equivalent of "inference infrastructure IQ" from Chapter 13 - one script
that produces a complete, versioned model package.
"""
import torch, torch.nn as nn
import numpy as np, pandas as pd
from torch.utils.data import Dataset, DataLoader
import sys, json, hashlib
sys.path.insert(0, '../krok6_bioreaktor_gru')
from symulacja_en import simulate_batch, kla, QO2_BASE, K_DO, DO_SAT, DT

torch.manual_seed(0)
FEATURES = ['stirring_rpm', 'aeration_vvm', 'feed_rate', 'X_biomass', 'DO']
W, H = 60, 30
MODEL_VERSION = "1.0.0"


def physics_rollout(df, t0, horizon=H):
    do = df['DO'].iloc[t0]
    for t in range(t0 + 1, t0 + horizon + 1):
        x = df['X_biomass'].iloc[t]
        otr = kla(df['stirring_rpm'].iloc[t - 1], df['aeration_vvm'].iloc[t - 1]) * (DO_SAT - do)
        our = QO2_BASE * x * do / (do + K_DO)
        do = np.clip(do + DT * (otr - our), 0.0, 100.0)
    return do


class WindowDataset(Dataset):
    def __init__(self, batches):
        self.samples, targets = [], []
        for df in batches:
            for t0 in range(W - 1, len(df) - H - 1):
                window = df[FEATURES].iloc[t0 - W + 1: t0 + 1].values
                target = df['DO'].iloc[t0 + H] - physics_rollout(df, t0)
                self.samples.append(window)
                targets.append(target)
        targets = np.array(targets, dtype=np.float32)
        self.m, self.s = float(targets.mean()), float(targets.std()) + 1e-6
        self.targets = (targets - self.m) / self.s

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return torch.tensor(self.samples[i], dtype=torch.float32), torch.tensor(self.targets[i], dtype=torch.float32)


class GRUPredictor(nn.Module):
    def __init__(self, n=5, h=16):
        super().__init__()
        self.gru = nn.GRU(n, h, batch_first=True)
        self.head = nn.Linear(h, 1)

    def forward(self, x):
        _, hn = self.gru(x)
        return self.head(hn.squeeze(0)).squeeze(-1)


if __name__ == "__main__":
    batches_train = [simulate_batch(seed=s) for s in range(12)]
    batches_calibration = [simulate_batch(seed=s) for s in range(12, 16)]

    ds = WindowDataset(batches_train)
    model = GRUPredictor()
    loader = DataLoader(ds, batch_size=64, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(20):
        for windows, targets in loader:
            opt.zero_grad()
            nn.MSELoss()(model(windows), targets).backward()
            opt.step()
    model.eval()
    print("Model trained.")

    # --- conformal prediction quantile (Chapter 10) ---
    def hybrid_prediction(df, t0i):
        x = torch.tensor(df[FEATURES].iloc[t0i - W + 1: t0i + 1].values, dtype=torch.float32).unsqueeze(0)
        pf = physics_rollout(df, t0i)
        with torch.no_grad():
            corr = model(x).item() * ds.s + ds.m
        return pf, corr

    calibration_errors = []
    for df in batches_calibration:
        for t0i in range(W - 1, len(df) - H - 1):
            pf, corr = hybrid_prediction(df, t0i)
            calibration_errors.append(abs(df['DO'].iloc[t0i + H] - (pf + corr)))
    calibration_errors = np.array(calibration_errors)
    quantile = float(np.sort(calibration_errors)[int(np.ceil((len(calibration_errors) + 1) * 0.90)) - 1])
    print(f"Conformal quantile (90%): {quantile:.3f}")

    # --- Mahalanobis statistics (OOD flag) ---
    all_train = pd.concat(batches_train)[FEATURES]
    mean_m = all_train.mean().values
    inv_cov = np.linalg.inv(np.cov(all_train.values.T))
    train_dist = np.array([np.sqrt((x - mean_m) @ inv_cov @ (x - mean_m)) for x in all_train.values])
    ood_threshold = float(np.percentile(train_dist, 99))
    print(f"OOD threshold (99th percentile): {ood_threshold:.3f}")

    # --- training data hash (data freeze, Chapter 8) ---
    everything = pd.concat(batches_train, keys=range(12))
    data_hash = hashlib.sha256(pd.util.hash_pandas_object(everything).values.tobytes()).hexdigest()[:16]

    # --- save all artifacts ---
    torch.save(model.state_dict(), "model_gru_en.pt")
    artifacts = {
        "model_version": MODEL_VERSION,
        "training_data_hash": data_hash,
        "features": FEATURES,
        "window_W": W,
        "horizon_H": H,
        "target_mean": ds.m,
        "target_std": ds.s,
        "conformal_quantile_90": quantile,
        "mahalanobis_mean": mean_m.tolist(),
        "mahalanobis_inverse_covariance": inv_cov.tolist(),
        "ood_threshold_99pct": ood_threshold,
    }
    with open("artefakty_en.json", "w") as f:
        json.dump(artifacts, f, indent=2)

    print("\nSaved: model_gru_en.pt, artefakty_en.json")

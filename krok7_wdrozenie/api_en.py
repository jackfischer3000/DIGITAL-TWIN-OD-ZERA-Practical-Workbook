"""
Step 7: an inference service as FastAPI - POST /predict endpoint.

Returns exactly what Chapter 11/13 of the book "Process Modeling in Pharma: From Zero to a Validated Model" specifies: a prediction with a confidence
interval, the OOD flag, the model version - plus the physics/ML-correction breakdown
(explainability, Chapter 12) and the fallback logic from the MDS.
"""
import json
import time
import sys
import os
from contextlib import asynccontextmanager

import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI
from pydantic import BaseModel, Field

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "krok6_bioreaktor_gru"))
from symulacja_en import kla, QO2_BASE, K_DO, DO_SAT, DT

with open(os.path.join(HERE, "artefakty_en.json")) as f:
    ART = json.load(f)

FEATURES = ART["features"]
W, H = ART["window_W"], ART["horizon_H"]
MEAN_M = np.array(ART["mahalanobis_mean"])
INV_COV = np.array(ART["mahalanobis_inverse_covariance"])
OOD_THRESHOLD = ART["ood_threshold_99pct"]
QUANTILE = ART["conformal_quantile_90"]


class GRUPredictor(nn.Module):
    def __init__(self, n=5, h=16):
        super().__init__()
        self.gru = nn.GRU(n, h, batch_first=True)
        self.head = nn.Linear(h, 1)

    def forward(self, x):
        _, hn = self.gru(x)
        return self.head(hn.squeeze(0)).squeeze(-1)


model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # loading weights AT STARTUP - not on every request (Chapter 13)
    global model
    model = GRUPredictor()
    model.load_state_dict(torch.load(os.path.join(HERE, "model_gru_en.pt")))
    model.eval()
    print(f"Model version {ART['model_version']} loaded (data: {ART['training_data_hash']})")
    yield


app = FastAPI(title="Bioreactor DO Prediction", lifespan=lifespan)


class Reading(BaseModel):
    stirring_rpm: float
    aeration_vvm: float
    feed_rate: float
    X_biomass: float
    DO: float


class PredictionRequest(BaseModel):
    window: list[Reading] = Field(..., description=f"The last {W} minutes of readings, oldest to newest")


class PredictionResponse(BaseModel):
    do_predicted: float
    interval_lower: float
    interval_upper: float
    physics_contribution: float
    ml_correction_contribution: float
    ood_flag: bool
    mahalanobis_distance: float
    mode: str
    model_version: str
    inference_time_ms: float


def physics_one_step(do_state, stirring, aeration, x_biomass):
    otr = kla(stirring, aeration) * (DO_SAT - do_state)
    our = QO2_BASE * x_biomass * do_state / (do_state + K_DO)
    return np.clip(do_state + DT * (otr - our), 0.0, 100.0)


def physics_rollout_from_window(window_df, horizon=H):
    do = window_df["DO"].iloc[-1]
    last_stirring = window_df["stirring_rpm"].iloc[-1]
    last_aeration = window_df["aeration_vvm"].iloc[-1]
    last_x = window_df["X_biomass"].iloc[-1]
    for _ in range(horizon):
        do = physics_one_step(do, last_stirring, last_aeration, last_x)
    return do


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    t0 = time.perf_counter()
    import pandas as pd
    window_df = pd.DataFrame([o.model_dump() for o in request.window])[FEATURES]

    # --- OOD flag: Mahalanobis distance of the last point ---
    last_point = window_df[FEATURES].iloc[-1].values
    diff = last_point - MEAN_M
    dist = float(np.sqrt(diff @ INV_COV @ diff))
    ood = dist > OOD_THRESHOLD

    # --- physics (always runs, even in fallback mode) ---
    pred_physics = float(physics_rollout_from_window(window_df))

    if ood:
        # FALLBACK PLAN from the MDS: physics-only mode, wider band
        return PredictionResponse(
            do_predicted=pred_physics,
            interval_lower=max(0.0, pred_physics - QUANTILE * 2),
            interval_upper=min(100.0, pred_physics + QUANTILE * 2),
            physics_contribution=pred_physics,
            ml_correction_contribution=0.0,
            ood_flag=True,
            mahalanobis_distance=dist,
            mode="FALLBACK (physics only)",
            model_version=ART["model_version"],
            inference_time_ms=(time.perf_counter() - t0) * 1000,
        )

    # --- normal mode: physics + GRU correction ---
    x = torch.tensor(window_df[FEATURES].values, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        correction = model(x).item() * ART["target_std"] + ART["target_mean"]
    prediction = pred_physics + correction

    return PredictionResponse(
        do_predicted=prediction,
        interval_lower=max(0.0, prediction - QUANTILE),
        interval_upper=min(100.0, prediction + QUANTILE),
        physics_contribution=pred_physics,
        ml_correction_contribution=correction,
        ood_flag=False,
        mahalanobis_distance=dist,
        mode="ACTIVE",
        model_version=ART["model_version"],
        inference_time_ms=(time.perf_counter() - t0) * 1000,
    )


@app.get("/health")
def health():
    return {"status": "ok", "model_version": ART["model_version"]}

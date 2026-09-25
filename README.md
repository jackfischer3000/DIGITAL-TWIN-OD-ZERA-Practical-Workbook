# Digital Twin From Scratch: A Practical Workbook

A companion workbook to *"Process Modeling in Pharma: From Zero to a Validated Model"* (Jacek Fischbach). It picks up exactly where the main book intentionally leaves off: the main book assumes you already know how to train a model and focuses on what happens after — DQ, OQ, PQ, documentation, deployment. This workbook builds that missing foundation, from the first line of code to a deployed, validated model.

**Available in both Polish and English.** The original workbook text and code comments are in Polish; an English translation — text, code, comments, and charts — now ships alongside it.

## What's inside

- **`Digital_Twin_od_zera_Practical_Workbook.docx`** — the full workbook in Polish (47 pages): runnable code, real results, and end-of-chapter exercises.
- **`Digital_Twin_from_Scratch_Workbook_EN.docx`** — the full English translation, same structure and page count.
- **`krok1_regresja/` … `krok5_pamiec/`** — Part I: ML fundamentals from scratch (linear regression → neural network → hybrid physics+ML model → overfitting/cross-validation → recurrence, on synthetic data with a known ground truth).
- **`krok6_bioreaktor_gru/`** — Part II: a full replica of the main book's bioreactor case study (dissolved oxygen prediction) — process simulation, a hybrid PyTorch/GRU architecture, Model Design Specification, a DQ review, OQ via conformal prediction, a PQ qualification report, drift monitoring and retraining, and a Model Risk Assessment.
- **`krok7_wdrozenie/`** — Part III: the trained model served as a FastAPI endpoint (`POST /predict`), with the OOD/fallback logic from the MDS actually wired up.
- **`krok8_roi/`** — a parameterized ROI calculator, verified against the main book's worked example, plus a sensitivity analysis.

Every `krokN_*/` folder has both a Polish script (`model.py`, `wizualizacja.py`, …) and its English counterpart (`model_en.py`, `wizualizacja_en.py`, …). The English scripts were run end-to-end to regenerate their own charts and results, not just translated in place — the numbers in the English workbook are the real output of the English code.

Every chapter's code is real and runnable — not pseudocode. The workbook keeps the mistakes that happened along the way (a neural network that extrapolated wildly, a model that trained for 20 epochs and learned nothing, a qualification run that failed and had to be diagnosed) because those teach more than a clean first-try success would.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install numpy pandas scikit-learn matplotlib torch fastapi uvicorn
```

Each `krokN_*/` folder is self-contained and runnable on its own (`python3 model.py`).

## Relation to the main book

| This workbook | Main book |
|---|---|
| Part I (fundamentals) | not covered — the main book assumes this knowledge |
| Chapter 6 (simulation, PyTorch, GRU) | Chapters 4–6 |
| Chapter 7 (MDS) | Chapter 7 |
| Chapter 8 (DQ) | Chapter 8 |
| Chapter 9 (OQ) | Chapter 10 |
| Chapter 10 (PQ) | Chapter 11 |
| Chapter 11 (monitoring, drift, retraining) | Chapter 14 |
| Chapter 12 (regulatory documentation, Model Risk Assessment) | Chapter 12 |
| Chapter 13 (deployment) | Chapter 13 |
| Chapter 14 (ROI) | Chapter 15 |

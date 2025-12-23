# Phoenix-Lite 0.1 (Prototype)

Phoenix-Lite is a lightweight computer-vision prototype for **real-time hand-signal recognition and session logging** using a webcam.  
It is an early MVP for a broader vision: privacy-aware, human-centered assistive AI for vulnerable users (including women with disabilities).

> Status: **Research / MVP prototype** (not production, not safety-critical)

---

## What it does

- Runs a **webcam pipeline** that tracks hands, predicts a gesture label, and shows a small **on-screen history** (last ~10s).
- Lets you **collect labelled gesture samples** from your own webcam (keyboard-assisted).
- Trains a **simple baseline ML model** from collected samples.
- Saves session artifacts (**JSON log + summary plot**) to a local folder for later review.

---

## Key features

- CLI workflow: `run`, `collect-gesture`, `train-gesture`.
- Gesture dataset collection as `.npz` samples (windowed features).
- Baseline training pipeline (StandardScaler + LogisticRegression).
- Session logging to `signals_tracked/` (JSON + graph).

---

## Quickstart

### 1) Install
This project uses **Poetry**.

```bash
poetry install
---

### 2) Run the app
poetry run phoenixlit run -c config.yaml

You will be prompted for a PIN (see auth section in the config).

---

### 3) Collect gesture samples
poetry run phoenixlit collect-gesture -c config.yaml

Keyboard mapping during collection (default):

1 = STOP

2 = YES

3 = NO

4 = PAIN

0 = NONE

SPACE = save current label (requires buffer to be full)

q or ESC = quit

Samples are saved to data/gesture/ by default.

---

### 4) Train a gesture model
poetry run phoenixlit train-gesture -c config.yaml

The model is saved to models/gesture.joblib by default.

---


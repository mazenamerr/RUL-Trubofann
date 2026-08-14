# RUL Prediction of Turbofan Engines

Remaining Useful Life (RUL) prediction for aircraft turbofan engines, built for the Intelligent Systems course. The project uses NASA's C-MAPSS **FD003** dataset (multiple operating conditions, two fault modes) and compares two deep learning architectures — a Transformer encoder and a CNN‑LSTM hybrid — for predicting how many operating cycles an engine has left before failure.

## Overview

Each engine in the dataset runs from a healthy state to failure while 21 sensors and 3 operational settings are recorded every cycle. The goal is to predict, from a rolling window of recent sensor readings, the number of cycles remaining before failure (RUL). This has direct value for predictive maintenance: scheduling servicing before failure, instead of on a fixed calendar or after breakdown.

Two label strategies were tested for every model:
- **Capped RUL** — RUL values are clipped at 150 cycles, since an engine far from failure looks identical from sensor data alone (a common simplification in RUL literature).
- **Uncapped RUL** — raw RUL values are used with no ceiling.

## Pipeline

The `src/` scripts run in numbered order:

1. **`1-load_and_eda.py`** — loads `train_FD003.csv` / `test_FD003.csv` / `RUL_FD003.csv`, checks engine counts and cycle ranges, plots example sensor traces.
2. **`2-make_rul.py`** — computes ground-truth RUL for the training set as `max_cycle_per_engine - current_cycle`, and inspects the RUL distribution.
3. **`3-prepare_data.py`** — selects sensor/setting feature columns, drops near-constant sensors (std ≤ 1e-3), and standardizes features (mean/std computed on train only) to produce `train_FD003_processed.csv` / `test_FD003_processed.csv`.
4. **`4-build_sequences.py`** — builds fixed-length sliding windows (`SEQ_LEN = 50` cycles) per engine, splitting engines 80/20 into train/validation, producing the `X_train_seq50.npy` / `y_train_seq50.npy` / `X_val_seq50.npy` / `y_val_seq50.npy` tensors.
5. **`5-transformer_model_{capped,uncapped}.py`** and **`5.2-CNN_LSTM_model_{capped,uncapped}.py`** — train the two architectures (details below) with early stopping on validation loss, using a weighted MSE loss that up-weights samples closer to failure (low RUL) so the models pay more attention to the safety-critical region.
6. **`6-test_plot_transf._{capped,uncapped}.py`** and **`6.2-test_plot_cnn_lstm_{capped,uncapped}.py`** — run the trained models on the FD003 test set (last window per engine), compute error metrics, save per-engine predictions, the 10 worst-predicted engines, and diagnostic plots (predicted vs. true, error histogram, absolute error vs. true RUL, CDF of absolute error).

## Models

**Transformer** — sensor/setting features are linearly projected to `d_model`, combined with a learned positional embedding, passed through a `TransformerEncoder` (1 layer, 4 attention heads), then mean-pooled and passed through a regression head to output a single RUL value per window.

**CNN‑LSTM** — a 1D convolutional layer extracts local temporal patterns across the sequence, followed by a multi-layer LSTM that models longer-range temporal dependencies, ending in a regression head.

Both models share: sequence length 50, weighted MSE loss (weight increases as true RUL approaches 0, capped at 2x), Adam optimizer, and early stopping (patience 8 epochs) on validation loss.

## Results (FD003 test set, 100 engines)

| Model | RUL strategy | MAE (cycles) | RMSE (cycles) |
|---|---|---|---|
| Transformer | Capped @150 | 11.85 | 16.95 |
| CNN-LSTM | Capped @150 | 12.53 | 17.84 |
| Transformer | Uncapped | 16.20 | 23.97 |
| CNN-LSTM | Uncapped | 16.04 | 24.28 |

Capping RUL at 150 cycles improved accuracy for both architectures, since it removes the ambiguity of predicting exact RUL far from failure (where sensor signals barely differ). The Transformer edged out the CNN-LSTM in the capped setting; the two architectures were close in the uncapped setting.

Diagnostic plots (predicted vs. true RUL, error histograms, absolute error vs. true RUL, error CDFs) for all four model/strategy combinations are in `plots/`.

## Repository structure

```
src/        Numbered pipeline scripts (EDA -> RUL labeling -> preprocessing -> sequencing -> training -> evaluation)
models/     Trained model weights (.pt) for all four model/strategy combinations
results/    Per-engine test predictions, worst-10 error cases, and feature scaler stats
plots/      Evaluation plots per model
```

## Data

The C-MAPSS FD003 dataset (train/test CSVs and RUL labels) is not included in this repo. It's publicly available from the [NASA Prognostics Data Repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/). Scripts expect the raw files (`train_FD003.csv`, `test_FD003.csv`, `RUL_FD003.csv`) under a `Data/` folder at the project root.

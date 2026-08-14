import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# -------------------- Plot styles --------------------
plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 9
})

# -------------------- Settings --------------------
SEQ_LEN = 50

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "Data")

test_processed_path = os.path.join(DATA_DIR, "test_FD003_processed.csv")
rul_test_path = os.path.join(DATA_DIR, "RUL_FD003.csv")

model_path = os.path.join(DATA_DIR, f"cnn_lstm_rul_uncapped_seq{SEQ_LEN}.pt")

# Outputs
plots_dir = os.path.join(DATA_DIR, "plots")
os.makedirs(plots_dir, exist_ok=True)
plots_cnn_lstm_dir = os.path.join(plots_dir, "CNN-LSTM")
os.makedirs(plots_cnn_lstm_dir, exist_ok=True)

eval_csv_path   = os.path.join(DATA_DIR, f"test_eval_cnn_lstm_uncapped_seq{SEQ_LEN}.csv")
worst_csv_path  = os.path.join(DATA_DIR, f"worst_10_engines_cnn_lstm_uncapped_seq{SEQ_LEN}.csv")
final_out       = os.path.join(DATA_DIR, f"final_test_results_cnn_lstm_uncapped_seq{SEQ_LEN}.csv")

scatter_path = os.path.join(plots_cnn_lstm_dir, f"pred_vs_true_cnn_lstm_uncapped_seq{SEQ_LEN}.png")
hist_path    = os.path.join(plots_cnn_lstm_dir, f"error_histogram_cnn_lstm_uncapped_seq{SEQ_LEN}.png")
abs_err_path = os.path.join(plots_cnn_lstm_dir, f"abs_error_vs_true_cnn_lstm_uncapped_seq{SEQ_LEN}.png")
cdf_path     = os.path.join(plots_cnn_lstm_dir, f"cdf_abs_error_cnn_lstm_uncapped_seq{SEQ_LEN}.png")

# ------------- Device --------------
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cuda")

print("Using device:", device)

# ---------- Model (Match training) ----------
print("\nLoading test data:", test_processed_path)
test_df = pd.read_csv(test_processed_path)
print("Test df shape:", test_df.shape)

# Use the same columns as pipeline
feature_cols = [c for c in test_df.columns if c not in ["engine_id", "cycle"]]
print("Num features:", len(feature_cols))

# Load true RUL labels (one per engine)
rul_df = pd.read_csv(rul_test_path)
if "RUL" in rul_df.columns:
    true_rul = rul_df["RUL"].values.astype(np.float32)
else:
    true_rul = rul_df.iloc[:, 0].values.astype(np.float32)

# -------------------- Build last sequence per engine --------------------
def build_last_sequences(df, seq_len, feature_cols):
    sequences = []
    engine_ids = []

    for eng_id, group in df.groupby("engine_id"):
        group = group.sort_values("cycle")
        data = group[feature_cols].values.astype(np.float32)

        if len(data) >= seq_len:
            seq = data[-seq_len:]  # last window
        else:
            pad_len = seq_len - len(data)
            pad = np.zeros((pad_len, data.shape[1]), dtype=np.float32)
            seq = np.vstack([pad, data]).astype(np.float32)

        sequences.append(seq)
        engine_ids.append(eng_id)

    X = np.stack(sequences, axis=0).astype(np.float32)
    ids = np.array(engine_ids)
    return X, ids

X_test, engine_ids = build_last_sequences(test_df, SEQ_LEN, feature_cols)
X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)

print("X_test:", X_test.shape, "| engines:", len(engine_ids))

# -------------------- CNN-LSTM model definition --------------------
class CNNLSTMRUL(nn.Module):
    def __init__(self, num_features, cnn_channels=32, kernel_size=3,
                 lstm_hidden=128, lstm_layers=2, dropout=0.2, use_mean_pool=False):
        super().__init__()
        padding = kernel_size // 2

        # CNN layer definition
        self.cnn = nn.Sequential(
            nn.Conv1d(num_features, cnn_channels, kernel_size, padding=padding),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),
            nn.Conv1d(cnn_channels, cnn_channels, kernel_size, padding=padding),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.lstm = nn.LSTM(
            input_size=cnn_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0
        )

        self.head = nn.Sequential(
            nn.LayerNorm(lstm_hidden),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden, 1)
        )

    def forward(self, x):
        x = x.transpose(1, 2)  # (B, F, T)
        z = self.cnn(x)        # (B, C, T)
        z = z.transpose(1, 2)  # (B, T, C)
        out, _ = self.lstm(z)  # (B, T, H)
        feat = out[:, -1, :]    # last timestep

        return self.head(feat).squeeze(-1)

# -------------------- Load checkpoint --------------------
print("\nLoading model:", model_path)
ckpt = torch.load(model_path, map_location=device)

num_features  = ckpt["num_features"]
cnn_channels  = ckpt.get("cnn_channels", 32)
kernel_size   = ckpt.get("kernel_size", 3)
lstm_hidden   = ckpt.get("lstm_hidden", 128)
lstm_layers   = ckpt.get("lstm_layers", 2)
dropout       = ckpt.get("dropout", 0.2)
rul_cap       = ckpt.get("rul_cap", None)

print("Checkpoint rul_cap:", rul_cap)

model = CNNLSTMRUL(
    num_features=num_features,
    cnn_channels=cnn_channels,
    kernel_size=kernel_size,
    lstm_hidden=lstm_hidden,
    lstm_layers=lstm_layers,
    dropout=dropout,
).to(device)

model.load_state_dict(ckpt["model_state"])
model.eval()

# -------------------- Predict --------------------
with torch.no_grad():
    preds = model(X_test_t).detach().cpu().numpy()

# -------------------- Align predictions with true RUL --------------------
pred_df = (
    pd.DataFrame({"engine_id": engine_ids, "pred_RUL": preds})
    .sort_values("engine_id")
    .reset_index(drop=True)
)

# True RUL file is in engine order (1..N). Ensure lengths match.
n = min(len(true_rul), len(pred_df))
pred_df = pred_df.head(n).assign(true_RUL=true_rul[:n])

pred_df["err"] = pred_df["pred_RUL"] - pred_df["true_RUL"]
pred_df["abs_err"] = pred_df["err"].abs()

rmse = float(np.sqrt(np.mean((pred_df["pred_RUL"] - pred_df["true_RUL"]) ** 2)))
mae  = float(np.mean(pred_df["abs_err"]))
bias = float(pred_df["err"].mean())

ss_res = ((pred_df["true_RUL"] - pred_df["pred_RUL"]) ** 2).sum()
ss_tot = ((pred_df["true_RUL"] - pred_df["true_RUL"].mean()) ** 2).sum()
r2 = 1 - ss_res / ss_tot

within_10 = (pred_df["abs_err"] <= 10).mean() * 100
within_20 = (pred_df["abs_err"] <= 20).mean() * 100

metric_text = f"MAE: {mae:.2f}\nRMSE: {rmse:.2f}\nBias: {bias:.2f}\nR² (Pred vs True) = {r2:.3f}\nN={len(pred_df)}"
print("\n" + metric_text)
print(f"% within ±10 cycles: {within_10:.2f}")
print(f"% within ±20 cycles: {within_20:.2f}")

# Save evaluation CSV
pred_df.to_csv(eval_csv_path, index=False)
print("Saved eval CSV:", eval_csv_path)

# Worst engines
worst10 = pred_df.sort_values("abs_err", ascending=False).head(10)
worst10.to_csv(worst_csv_path, index=False)
print("Saved worst-10 CSV:", worst_csv_path)

# -------------------- Plot 1: Pred vs True scatter --------------------
plt.figure()
plt.scatter(pred_df["true_RUL"], pred_df["pred_RUL"], s=18, alpha=0.7, color='tab:green', label="Model predictions")
plt.grid(True, alpha=0.2)
plt.xlabel("True RUL")
plt.ylabel("Predicted RUL")
plt.title("Predicted vs True RUL (CNN-LSTM Uncapped Model)", fontweight="bold")

mmin = min(pred_df["true_RUL"].min(), pred_df["pred_RUL"].min())
mmax = max(pred_df["true_RUL"].max(), pred_df["pred_RUL"].max())
plt.plot([mmin, mmax], [mmin, mmax], linestyle="-", color='tab:red', label="Perfect prediction (y = x)")
plt.legend()

plt.text(
    0.02, 0.95, metric_text,
    transform=plt.gca().transAxes,
    va="top",
    fontsize=9,
    bbox=dict(boxstyle="round", alpha=0.2)
)

plt.savefig(scatter_path, dpi=200, bbox_inches="tight")
plt.close()
print("Saved:", scatter_path)

# -------------------- Plot 2: Error histogram --------------------
plt.figure()
plt.hist(pred_df["err"], bins=25, edgecolor="black", color='green', linewidth=0.8, label="Prediction error")
plt.grid(True, alpha=0.2)
plt.xlabel("Prediction Error (Pred - True)")
plt.ylabel("Number of engines")
plt.title("Distribution of Pred. Err. (CNN-LSTM Uncapped Model)", fontweight="bold")
plt.axvline(0, color="black", linewidth=2, label="Zero error")
plt.axvline(pred_df["err"].mean(), color="red", linewidth=2, linestyle="--", label="Mean error (bias)")
plt.legend()

for bar in plt.gca().patches:
    h = bar.get_height()
    if h > 0:
        plt.text(bar.get_x() + bar.get_width()/2, h, f"{h:.0f}",
                 ha="center", va="bottom", fontsize=9)

plt.text(
    0.66, 0.55, metric_text,
    transform=plt.gca().transAxes,
    va="center",
    fontsize=9,
    bbox=dict(boxstyle="round", alpha=0.2)
)

plt.savefig(hist_path, dpi=200, bbox_inches="tight")
plt.close()
print("Saved:", hist_path)

# -------------------- Plot 3: Abs error vs True RUL --------------------
plt.figure()
plt.scatter(pred_df["true_RUL"], pred_df["abs_err"], s=16, alpha=0.7, label="Absolute error per engine", color='tab:green')
plt.grid(True, alpha=0.2)
plt.xlabel("True RUL")
plt.ylabel("Absolute Error")
plt.title("Absolute Error vs True RUL (CNN-LSTM Uncapped Model)", fontweight="bold")

plt.axhline(pred_df["abs_err"].mean(), color='r', linestyle="--", linewidth=1, label="MAE level")
plt.legend()

plt.savefig(abs_err_path, dpi=200, bbox_inches="tight")
plt.close()
print("Saved:", abs_err_path)

# -------------------- Plot 4: CDF of absolute error --------------------
abs_err_sorted = np.sort(pred_df["abs_err"].values)
cdf = np.arange(1, len(abs_err_sorted) + 1) / len(abs_err_sorted)

plt.figure()
plt.plot(abs_err_sorted, cdf, linewidth=2, color='green')
plt.grid(True, alpha=0.2)
plt.xlabel("Absolute Error")
plt.ylabel("CDF")
plt.title("CDF of Absolute Pred. Err. (CNN-LSTM Uncapped Model)", fontweight="bold")

plt.axvline(10, linestyle="-", linewidth=1, color='r', label="Error ≤ 10 cycles (Vertical)")
plt.axvline(20, linestyle="--", linewidth=1, color='r', label="Error ≤ 20 cycles (Vertical)")
plt.axhline(0.5, linestyle="-", linewidth=1, color='orange', label="50% of engines (Horizontal)")
plt.axhline(0.8, linestyle="--", linewidth=1, color='orange', label="80% of engines (Horizontal)")
plt.legend()

plt.savefig(cdf_path, dpi=200, bbox_inches="tight")
plt.close()
print("Saved:", cdf_path)

pred_df[["engine_id", "true_RUL", "pred_RUL", "err", "abs_err"]].to_csv(final_out, index=False)
print("\nSaved final results CSV to:", final_out)

print("\nDone.")

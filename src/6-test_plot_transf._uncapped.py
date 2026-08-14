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

model_path = os.path.join(DATA_DIR, f"transformer_rul_uncapped_seq{SEQ_LEN}.pt")

plots_dir = os.path.join(DATA_DIR, "plots")
os.makedirs(plots_dir, exist_ok=True)
plots_transformer_dir = os.path.join(plots_dir, "Transformer")
os.makedirs(plots_transformer_dir, exist_ok=True)

eval_csv_path = os.path.join(DATA_DIR, f"test_eval_transformer_uncapped_seq{SEQ_LEN}.csv")
worst_csv_path = os.path.join(DATA_DIR, f"worst_10_transformer_uncapped_seq{SEQ_LEN}.csv")
final_out = os.path.join(DATA_DIR, f"final_test_results_transformer_uncapped_seq{SEQ_LEN}.csv")

# --------------- Device ----------------
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cuda")

print("Using device:", device)

# ---------- Model (Match training) ----------
class TransformerRUL(nn.Module):
    def __init__(self, num_features, seq_len, d_model, nhead, num_layers, dropout):
        super().__init__()

        self.input_proj = nn.Linear(num_features, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, seq_len, d_model))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="relu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.attn_pool = nn.Linear(d_model, 1)

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1),
        )

    def forward(self, x):
        x = self.input_proj(x)
        x = x + self.pos_emb[:, : x.size(1)]
        x = self.encoder(x)

        # Attention pooling
        scores = self.attn_pool(x).squeeze(-1)     # (B, T)
        weights = torch.softmax(scores, dim=1)     # (B, T)
        x = (x * weights.unsqueeze(-1)).sum(dim=1) # (B, D)

        y = self.head(x).squeeze(1)
        return y

# ---------- Build last sequence per engine ----------
def build_last_sequences(test_df, seq_len, feature_cols):
    sequences, engine_ids = [], []

    for eng_id, group in test_df.groupby("engine_id"):
        group = group.sort_values("cycle")
        feat = group[feature_cols].values

        if len(feat) < seq_len:
            pad_len = seq_len - len(feat)
            pad = np.repeat(feat[:1], pad_len, axis=0)
            feat = np.vstack([pad, feat])

        sequences.append(feat[-seq_len:])
        engine_ids.append(eng_id)

    X = np.array(sequences, dtype=np.float32)
    return X, engine_ids

# ---------- Load test data ----------
test_df = pd.read_csv(test_processed_path)
print("Loaded test:", test_df.shape)

# Identify feature columns
feature_cols = [c for c in test_df.columns if c not in ["engine_id", "cycle"]]
print("Num features:", len(feature_cols))

# Load true RUL per engine
rul_df = pd.read_csv(rul_test_path)
true_rul = rul_df["RUL"].values

# Build one sample per engine (last window)
X_test, engine_ids = build_last_sequences(test_df, SEQ_LEN, feature_cols)
X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)

# ---------- Load model checkpoint ----------
ckpt = torch.load(model_path, map_location=device)

model = TransformerRUL(
    num_features=ckpt["num_features"],
    seq_len=ckpt["seq_len"],
    d_model=ckpt["d_model"],
    nhead=ckpt["nhead"],
    num_layers=ckpt["num_layers"],
    dropout=ckpt["dropout"],
).to(device)

model.load_state_dict(ckpt["model_state"])
model.eval()

# ---------- Predict ----------
with torch.no_grad():
    preds = model(X_test_t).detach().cpu().numpy()

# ---------- Align true RUL with engine ids ----------
pred_df = (
    pd.DataFrame({"engine_id": engine_ids, "pred_RUL": preds})
    .sort_values("engine_id")
    .reset_index(drop=True)
)

n = min(len(true_rul), len(pred_df))
pred_df = pred_df.head(n).assign(true_RUL=true_rul[:n])

# ---------- Metrics ----------
print("\n=== Metrics + diagnostics ===")
pred_df["err"] = pred_df["pred_RUL"] - pred_df["true_RUL"]
pred_df["abs_err"] = np.abs(pred_df["err"])
pred_df["sq_err"] = pred_df["err"] ** 2

mae = float(pred_df["abs_err"].mean())
rmse = float(np.sqrt(pred_df["sq_err"].mean()))
bias = float(pred_df["err"].mean())

ss_res = ((pred_df["true_RUL"] - pred_df["pred_RUL"]) ** 2).sum()
ss_tot = ((pred_df["true_RUL"] - pred_df["true_RUL"].mean()) ** 2).sum()
r2 = 1 - ss_res / ss_tot

abs_err = np.sort(pred_df["abs_err"]) 
cdf = np.arange(1, len(abs_err)+1) / len(abs_err)

within_10 = (pred_df["abs_err"] <= 10).mean() * 100
within_20 = (pred_df["abs_err"] <= 20).mean() * 100

print(f"\nMAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"Bias: {bias:.2f}")
print(f"R² Pred vs True): {r2:.2f}")
print(f"% within ±10 cycles: {within_10:.2f}")
print(f"% within ±20 cycles: {within_20:.2f}")

# Save evaluation CSV
pred_df.to_csv(eval_csv_path, index=False)
print("Saved eval CSV:", eval_csv_path)

# Worst engines
worst10 = pred_df.sort_values("abs_err", ascending=False).head(10)
worst10.to_csv(worst_csv_path, index=False)
print("Saved worst-10 CSV:", worst_csv_path)

# ---------- Plots ----------
print("\n=== Saving plots ===")
metric_text = f"MAE = {mae:.2f}\nRMSE = {rmse:.2f}\nBias = {bias:.2f}\nR² (Pred vs True) = {r2:.3f}"

# Plot A: True vs Pred scatter
plt.figure()
plt.scatter(pred_df["true_RUL"], pred_df["pred_RUL"], s=18, alpha=0.8, label="Model predictions")
min_v = float(min(pred_df["true_RUL"].min(), pred_df["pred_RUL"].min()))
max_v = float(max(pred_df["true_RUL"].max(), pred_df["pred_RUL"].max()))
plt.plot([min_v, max_v], [min_v, max_v], linewidth=1, color='r', label="Perfect prediction (y = x)")  # diagonal reference
plt.grid(True, alpha=0.2)
plt.xlabel("True RUL")
plt.ylabel("Predicted RUL")
plt.title("Predicted vs True RUL (Transf. Uncapped Model)", fontweight="bold")
plt.text(0.05, 0.95, metric_text, transform=plt.gca().transAxes, va="top", fontsize=9, bbox=dict(boxstyle="round", alpha=0.15))
plt.legend()
scatter_path = os.path.join(plots_transformer_dir, f"true_vs_pred_scatter_transf_uncapped_seq{SEQ_LEN}.png")
plt.savefig(scatter_path, dpi=200, bbox_inches="tight")
plt.close()
print("Saved:", scatter_path)

# Plot B: Error histogram
plt.figure()
plt.hist(pred_df["err"], bins=25, edgecolor="black", linewidth=0.8, label="Prediction error")
plt.grid(True, linestyle="-", alpha=0.2)
plt.xlabel("Prediction Error (Pred - True)")
plt.ylabel("Number of engines")
plt.title("Distribution of Pred. Err. (Transf. Uncapped Model)", fontweight="bold")
plt.axvline(0, color="black", linewidth=2, label="Zero error")
plt.axvline(pred_df["err"].mean(), color="red", linewidth=2, linestyle="--", label="Mean error (bias)")
plt.legend()
for bar in plt.gca().patches:
    h = bar.get_height()
    if h > 0:
        plt.text(bar.get_x() + bar.get_width()/2, h, f"{h:.0f}",
                 ha="center", va="bottom", fontsize=9)
plt.text(
    0.65, 0.55, metric_text,
    transform=plt.gca().transAxes,
    va="center",
    fontsize=9,
    bbox=dict(boxstyle="round", alpha=0.2)
)
hist_path = os.path.join(plots_transformer_dir, f"error_histogram_transf_uncapped_seq{SEQ_LEN}.png")
plt.savefig(hist_path, dpi=200, bbox_inches="tight")
plt.close()
print("Saved:", hist_path)

# Plot C: Absolute error vs True RUL
plt.figure()
plt.scatter(pred_df["true_RUL"], pred_df["abs_err"], s=16, alpha=0.7, label="Absolute error per engine")
plt.grid(True, alpha=0.2)
plt.xlabel("True RUL")
plt.ylabel("Absolute Error")
plt.title("Absolute Error vs True RUL (Transf. Uncapped Model)", fontweight="bold")
plt.axhline(pred_df["abs_err"].mean(), linestyle="--", color='r', linewidth=1, label="MAE level")
plt.legend()
abs_err_path = os.path.join(plots_transformer_dir, f"abs_error_vs_true_transf_uncapped_seq{SEQ_LEN}.png")
plt.savefig(abs_err_path, dpi=200, bbox_inches="tight")
plt.close()
print("Saved:", abs_err_path)

# -------------------- Plot 4: CDF of absolute error --------------------
plt.figure()
plt.plot(abs_err, cdf, linewidth=2)
plt.grid(True, linestyle="--", alpha=0.2)
plt.xlabel("Absolute Error (cycles)")
plt.ylabel("Fraction of engines")
plt.title("CDF of Absolute Pred. Err. (Transf. Uncapped Model)", fontweight="bold")
plt.axvline(10, linestyle="-", linewidth=1, color='r', label="Error ≤ 10 cycles (Vertical)")
plt.axvline(20, linestyle="--", linewidth=1, color='r', label="Error ≤ 20 cycles (Vertical)")
plt.axhline(0.5, linestyle="-", linewidth=0.8, color='orange', label="50%' of engines (Horizontal)")
plt.axhline(0.8, linestyle="--", linewidth=0.8, color='orange', label="80%' of engines (Horizontal)")
plt.legend()
cdf_path = os.path.join(plots_transformer_dir, f"cdf_abs_error_transf_uncapped_seq{SEQ_LEN}.png")
plt.savefig(cdf_path, dpi=200, bbox_inches="tight")
plt.close()
print("Saved:", cdf_path)

# ---------- Save final results ----------
pred_df[["engine_id", "true_RUL", "pred_RUL", "err", "abs_err"]].to_csv(final_out, index=False)
print("\nSaved final results CSV to:", final_out)

print("\nDone.")

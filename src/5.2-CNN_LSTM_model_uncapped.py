import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ---------- 1) Settings ----------
SEQ_LEN, BATCH_SIZE, EPOCHS = 50, 256, 100
LR, DROPOUT = 4e-4, 0.2
RUL_CAP, PATIENCE = None, 8
RUL_WEIGHT_REF = 150.0

CNN_CHANNELS = 64
KERNEL_SIZE = 3
LSTM_HIDDEN = 256
LSTM_LAYERS = 2

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "Data")

X_train_path = os.path.join(DATA_DIR, f"X_train_seq{SEQ_LEN}.npy")
y_train_path = os.path.join(DATA_DIR, f"y_train_seq{SEQ_LEN}.npy")
X_val_path   = os.path.join(DATA_DIR, f"X_val_seq{SEQ_LEN}.npy")
y_val_path   = os.path.join(DATA_DIR, f"y_val_seq{SEQ_LEN}.npy")

model_out_path = os.path.join(DATA_DIR, f"cnn_lstm_rul_uncapped_seq{SEQ_LEN}.pt")

# ---------- 2) Device ----------
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cuda")

print("Using device:", device)

# ---------- 3) Load sequences ----------
print("\nLoading sequences...")
X_train, y_train = np.load(X_train_path), np.load(y_train_path)
X_val,   y_val   = np.load(X_val_path),   np.load(y_val_path)

y_train = np.clip(y_train, 0, None).astype(np.float32)
y_val   = np.clip(y_val,   0, None).astype(np.float32)

print("X_train:", X_train.shape, "y_train:", y_train.shape, "y_train min/max:", float(y_train.min()), float(y_train.max()))
print("X_val  :", X_val.shape,   "y_val  :", y_val.shape,   "y_val min/max:", float(y_val.min()), float(y_val.max()))

num_features = X_train.shape[2]
print("num_features:", num_features)

# ---------- 4) Weights (uncapped-friendly) ----------
def make_weights(y: np.ndarray, ref: float = 150.0):
    y = y.astype(np.float32)
    w = 1.0 + np.maximum((ref - y) / ref, 0.0)
    return np.clip(w, 1.0, 2.0).astype(np.float32)

w_train = make_weights(y_train, ref=RUL_WEIGHT_REF)
w_val = np.ones_like(y_val, dtype=np.float32)

print("w_train min/max:", float(w_train.min()), float(w_train.max()))

# ---------- 5) Dataset / Dataloader ----------
class SeqDataset(Dataset):
    def __init__(self, X, y, w):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.w = torch.tensor(w, dtype=torch.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.w[idx]

train_loader = DataLoader(
    SeqDataset(X_train, y_train, w_train),
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=False
)
val_loader = DataLoader(
    SeqDataset(X_val, y_val, w_val),
    batch_size=BATCH_SIZE,
    shuffle=False,
    drop_last=False
)

# ---------- 6) CNN-LSTM model ----------
class CNNLSTMRUL(nn.Module):
    def __init__(self, num_features, cnn_channels=64, kernel_size=3,
                 lstm_hidden=128, lstm_layers=2, dropout=0.1):
        super().__init__()

        padding = kernel_size // 2

        # Conv1d (B, C_in, T)
        self.cnn = nn.Sequential(
            nn.Conv1d(num_features, cnn_channels, kernel_size, padding=padding),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),
            nn.Conv1d(cnn_channels, cnn_channels, kernel_size, padding=padding),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # LSTM (B, T, C)
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
        # x: (B, T, F)
        x = x.transpose(1, 2)          # (B, F, T) for Conv1d
        z = self.cnn(x)                # (B, C, T)
        z = z.transpose(1, 2)          # (B, T, C) for LSTM
        out, _ = self.lstm(z)          # (B, T, H)
        feat = out[:, -1, :]           # last timestep
        return self.head(feat).squeeze(-1)

model = CNNLSTMRUL(
    num_features=num_features,
    cnn_channels=CNN_CHANNELS,
    kernel_size=KERNEL_SIZE,
    lstm_hidden=LSTM_HIDDEN,
    lstm_layers=LSTM_LAYERS,
    dropout=DROPOUT
).to(device)

# ---------- 7) Metrics ----------
def rmse(pred, target):
    return torch.sqrt(torch.mean((pred - target) ** 2))

def mae(pred, target):
    return torch.mean(torch.abs(pred - target))

base_loss = nn.SmoothL1Loss(beta=10.0, reduction="none")

# Optimizer / scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-5
)

# ---------- 8) Training loop ----------
best_val_rmse = float("inf")
best_val_mae = float("inf")
bad_epochs = 0

print("\nTraining CNN-LSTM (UNCAPPED)...")

for epoch in range(1, EPOCHS + 1):
    # ---- Train ----
    model.train()
    train_loss_sum = 0.0
    train_rmse_sum = 0.0
    train_mae_sum  = 0.0
    n_train = 0

    for xb, yb, wb in train_loader:
        xb = xb.to(device)
        yb = yb.to(device)
        wb = wb.to(device)

        optimizer.zero_grad()

        pred = model(xb)
        pred = torch.clamp(pred, min=0.0)

        loss_vec = base_loss(pred, yb)      # (B,)
        loss = (loss_vec * wb).mean()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        bs = xb.size(0)
        train_loss_sum += loss.item() * bs
        train_rmse_sum += rmse(pred.detach(), yb).item() * bs
        train_mae_sum  += mae(pred.detach(), yb).item() * bs
        n_train += bs

    train_loss = train_loss_sum / n_train
    train_rmse_ = train_rmse_sum / n_train
    train_mae_  = train_mae_sum / n_train

    # ---- Val ----
    model.eval()
    val_loss_sum = 0.0
    val_rmse_sum = 0.0
    val_mae_sum  = 0.0
    n_val = 0

    with torch.no_grad():
        for xb, yb, wb in val_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            wb = wb.to(device)

            pred = model(xb)
            pred = torch.clamp(pred, min=0.0)

            loss_vec = base_loss(pred, yb)
            loss = (loss_vec * wb).mean()

            bs = xb.size(0)
            val_loss_sum += loss.item() * bs
            val_rmse_sum += rmse(pred, yb).item() * bs
            val_mae_sum  += mae(pred, yb).item() * bs
            n_val += bs

    val_loss = val_loss_sum / n_val
    val_rmse_ = val_rmse_sum / n_val
    val_mae_  = val_mae_sum / n_val

    print(
        f"Epoch {epoch}/{EPOCHS}"
        f" | train_loss={train_loss:.3f} train_rmse={train_rmse_:.3f} train_mae={train_mae_:.3f}"
        f" | val_loss={val_loss:.3f} val_rmse={val_rmse_:.3f} val_mae={val_mae_:.3f}"
    )

    scheduler.step(val_rmse_)

    if val_mae_ < best_val_mae:
        best_val_mae = val_mae_

    if val_rmse_ < best_val_rmse:
        best_val_rmse = val_rmse_
        bad_epochs = 0

        torch.save(
            {
                "model_state": model.state_dict(),
                "num_features": num_features,
                "seq_len": SEQ_LEN,
                "cnn_channels": CNN_CHANNELS,
                "kernel_size": KERNEL_SIZE,
                "lstm_hidden": LSTM_HIDDEN,
                "lstm_layers": LSTM_LAYERS,
                "dropout": DROPOUT,
                "rul_cap": None,  # uncapped
                "rul_weight_ref": float(RUL_WEIGHT_REF),
            },
            model_out_path
        )
        print("Saved new best model (by val RMSE).")
    else:
        bad_epochs += 1
        if bad_epochs >= PATIENCE:
            print(f"Early stopping at epoch {epoch} (no val RMSE improvement for {PATIENCE} epochs).")
            break

print(f"\nTraining finished. Best val MAE: {best_val_mae:.3f} Best val RMSE: {best_val_rmse:.3f}")
print("Model saved at:", model_out_path)

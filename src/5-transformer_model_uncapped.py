import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ---------- 1) Settings ----------
SEQ_LEN, BATCH_SIZE, EPOCHS = 50, 256, 100
LR, D_MODEL, NHEAD, NUM_LAYERS, DROPOUT = 5e-4, 64, 4, 1, 0.2
RUL_CAP, PATIENCE = None, 8 
RUL_WEIGHT_REF = 150.0

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "Data")

X_train_path = os.path.join(DATA_DIR, f"X_train_seq{SEQ_LEN}.npy")
y_train_path = os.path.join(DATA_DIR, f"y_train_seq{SEQ_LEN}.npy")
X_val_path   = os.path.join(DATA_DIR, f"X_val_seq{SEQ_LEN}.npy")
y_val_path   = os.path.join(DATA_DIR, f"y_val_seq{SEQ_LEN}.npy")

model_out_path = os.path.join(DATA_DIR, f"transformer_rul_uncapped_seq{SEQ_LEN}.pt")

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

# We only ensure non-negative RUL. We do not cap RUL.
y_train = np.clip(y_train, 0, None).astype(np.float32)
y_val   = np.clip(y_val,   0, None).astype(np.float32)

print("y_train min/max:", y_train.min(), y_train.max())
print("y_val   min/max:", y_val.min(), y_val.max())
print("X_train shape:", X_train.shape, "y_train shape:", y_train.shape)
print("X_val   shape:", X_val.shape, "y_val   shape:", y_val.shape)

num_features = X_train.shape[-1]
print("Number of features:", num_features)

# ---------- 4) Weighted loss helpers ----------
def make_weights(y: np.ndarray, ref: float = 150.0):
    y = y.astype(np.float32)
    w = 1.0 + np.maximum((ref - y) / ref, 0.0)
    return np.clip(w, 1.0, 2.0).astype(np.float32)

w_train = make_weights(y_train, ref=RUL_WEIGHT_REF)
w_val   = np.ones_like(y_val, dtype=np.float32)

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

# ---------- 6) Transformer model ----------
class TransformerRUL(nn.Module):
    def __init__(self, num_features, seq_len, d_model, nhead, num_layers, dropout):
        super().__init__()

        self.input_proj = nn.Linear(num_features, d_model)

        # learned positional embedding
        # Transformers do not inherently know the time order, so we add a position vector per timestep.
        self.pos_emb = nn.Parameter(torch.zeros(1, seq_len, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="relu"
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # attention pooling:
        # produces a learned weighted average across all timesteps, so the model can focus on important cycles.
        self.attn_pool = nn.Linear(d_model, 1)

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1)
        )

    def forward(self, x):
        x = self.input_proj(x)                 # (B, T, D)
        x = x + self.pos_emb[:, : x.size(1)]   # (B, T, D)
        x = self.encoder(x)                    # (B, T, D)

        # attention pooling
        # produces a learned weighted average across all timesteps, so the model can focus on important cycles.
        scores = self.attn_pool(x).squeeze(-1)       # (B, T)
        weights = torch.softmax(scores, dim=1)       # (B, T)
        x = (x * weights.unsqueeze(-1)).sum(dim=1)   # (B, D)

        y = self.head(x).squeeze(1)                  # (B,)
        return y

model = TransformerRUL(
    num_features=num_features,
    seq_len=SEQ_LEN,
    d_model=D_MODEL,
    nhead=NHEAD,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT
).to(device)

# ---------- 7) Metrics ----------
def rmse(pred, target):
    return torch.sqrt(torch.mean((pred - target) ** 2))

def mae(pred, target):
    return torch.mean(torch.abs(pred - target))

# Weighted SmoothL1 (Huber) loss per sample, Good for validity to outliers compared to RMSE
base_loss = nn.SmoothL1Loss(beta=10.0, reduction="none")

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-5
)

best_val_rmse = float("inf")
best_val_mae = float("inf")
bad_epochs = 0

# ---------- 8) Train loop ----------
print("\nTraining Transformer Uncapped...")
for epoch in range(1, EPOCHS + 1):
    # ------ train ------
    model.train()
    train_loss_sum, train_rmse_sum, train_mae_sum, n_train = 0.0, 0.0, 0.0, 0

    for xb, yb, wb in train_loader:
        xb, yb, wb = xb.to(device), yb.to(device), wb.to(device)

        optimizer.zero_grad()

        # Uncapped predictions
        pred = model(xb)
        pred = torch.clamp(pred, min=0.0)

        loss = (base_loss(pred, yb) * wb).mean()

        loss.backward()

        # Gradient clipping improves stability for transformer training, preventing gradients exploding
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

    # ------ validation ------
    model.eval()
    val_loss_sum, val_rmse_sum, val_mae_sum, n_val = 0.0, 0.0, 0.0, 0

    with torch.no_grad():
        for xb, yb, wb in val_loader:
            xb, yb, wb = xb.to(device), yb.to(device), wb.to(device)

            pred = model(xb)
            pred = torch.clamp(pred, min=0.0)

            loss = (base_loss(pred, yb) * wb).mean()

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

    if val_rmse_ < best_val_rmse:  # Save best + early stop
        best_val_rmse = val_rmse_
        bad_epochs = 0
        if val_mae_ < best_val_mae:
            best_val_mae = val_mae_
            
        torch.save(
            {
                "model_state": model.state_dict(),
                "num_features": num_features,
                "seq_len": SEQ_LEN,
                "d_model": D_MODEL,
                "nhead": NHEAD,
                "num_layers": NUM_LAYERS,
                "dropout": DROPOUT,
                "rul_cap": None,              
                "rul_weight_ref": RUL_WEIGHT_REF,
            },
            model_out_path
        )
        print("Best Result Values\n")
    else:
        bad_epochs += 1
        if bad_epochs >= PATIENCE:
            print(f"Early stopping at epoch {epoch} (no val improvement for {PATIENCE} epochs).")
            break

print(f"\nTraining finished. Best val MAE: {best_val_mae:.3f} Best val RMSE: {best_val_rmse:.3f}")
print("Model saved at:", model_out_path)

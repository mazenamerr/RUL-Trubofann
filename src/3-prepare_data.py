import os
import pandas as pd

# ---------- 1. File paths ----------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "Data")

train_with_rul_path = os.path.join(DATA_DIR, "train_FD003_with_RUL.csv")
test_path = os.path.join(DATA_DIR, "test_FD003.csv")

train_out_path = os.path.join(DATA_DIR, "train_FD003_processed.csv")
test_out_path = os.path.join(DATA_DIR, "test_FD003_processed.csv")
scaler_out_path = os.path.join(DATA_DIR, "feature_scaler_stats.csv")

# ---------- 2. Load data ----------
print("Loading data...")

train_df = pd.read_csv(train_with_rul_path)
test_df = pd.read_csv(test_path)

print("Train shape (with RUL):", train_df.shape)
print("Test shape            :", test_df.shape)
print("Train columns:", train_df.columns.tolist())

# ---------- 3. Choose feature columns ----------
# We keep only settings + sensor columns as features
feature_cols = [c for c in train_df.columns
                if c.startswith("setting") or c.startswith("s")]

print("\nFeature columns (before dropping low-variance):")
print(feature_cols)


assert "engine_id" in train_df.columns
assert "cycle" in train_df.columns
assert "RUL" in train_df.columns

# ---------- 4. Drop near-constant features ----------

feature_stds = train_df[feature_cols].std()


threshold = 1e-3
kept_features = feature_stds[feature_stds > threshold].index.tolist()
dropped_features = feature_stds[feature_stds <= threshold].index.tolist()

print("\nStandard deviation of features:")
print(feature_stds)

print("\nDropped features (almost constant):", dropped_features)
print("Kept features:", kept_features)

train_feats = train_df[kept_features].copy()
test_feats = test_df[kept_features].copy()

# ---------- 5. Compute mean and std on training features ----------
feature_means = train_feats.mean()
feature_stds = train_feats.std()

print("\nFeature means (train):")
print(feature_means)

print("\nFeature stds (train):")
print(feature_stds)

# ---------- 6. Normalize features ----------
# x_norm = (x - mean) / std (using training stats)
train_norm = (train_feats - feature_means) / feature_stds
test_norm = (test_feats - feature_means) / feature_stds

# ---------- 7. Build processed train & test DataFrames ----------
train_processed = pd.concat(
    [
        train_df[["engine_id", "cycle", "RUL"]].reset_index(drop=True),
        train_norm.reset_index(drop=True)
    ],
    axis=1
)

test_processed = pd.concat(
    [
        test_df[["engine_id", "cycle"]].reset_index(drop=True),
        test_norm.reset_index(drop=True)
    ],
    axis=1
)

print("\nProcessed train shape:", train_processed.shape)
print("Processed test shape :", test_processed.shape)

print("\nSample of processed training data:")
print(train_processed.head())

# ---------- 8. Save processed data ----------
train_processed.to_csv(train_out_path, index=False)
test_processed.to_csv(test_out_path, index=False)

print(f"\nSaved processed train to: {train_out_path}")
print(f"Saved processed test  to: {test_out_path}")

# ---------- 9. Save scaler stats (for your friend / model) ----------
scaler_stats = pd.DataFrame({
    "feature": kept_features,
    "mean": feature_means.values,
    "std": feature_stds.values
})

scaler_stats.to_csv(scaler_out_path, index=False)
print(f"Saved feature scaler stats to: {scaler_out_path}")

print("\nDone.")

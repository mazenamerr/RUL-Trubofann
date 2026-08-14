import os
import numpy as np
import pandas as pd

# ---------- 1. Settings ----------
SEQ_LEN = 50  # length of history window in cycles

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "Data")

train_processed_path = os.path.join(DATA_DIR, "train_FD003_processed.csv")

# Output .npy files 
X_train_out = os.path.join(DATA_DIR, f"X_train_seq{SEQ_LEN}.npy")
y_train_out = os.path.join(DATA_DIR, f"y_train_seq{SEQ_LEN}.npy")
X_val_out   = os.path.join(DATA_DIR, f"X_val_seq{SEQ_LEN}.npy")
y_val_out   = os.path.join(DATA_DIR, f"y_val_seq{SEQ_LEN}.npy")

# ---------- 2. Load processed training data ----------
print("Loading processed training data...")
df = pd.read_csv(train_processed_path)
print("Data shape:", df.shape)
print("Columns:", df.columns.tolist())

# Feature columns = everything except engine_id, cycle, RUL
feature_cols = [c for c in df.columns if c not in ["engine_id", "cycle", "RUL"]]
num_features = len(feature_cols)
print("\nUsing feature columns:", feature_cols)
print("Number of features:", num_features)

# ---------- 3. Split engines into train / validation ----------
engine_ids = sorted(df["engine_id"].unique())
print("\nAll engine IDs:", engine_ids)

# Use first 80 engines for training, last 20 for validation
train_engine_ids = engine_ids[:-20]
val_engine_ids = engine_ids[-20:]

print("Train engine IDs (first 5):", train_engine_ids[:5], "...", train_engine_ids[-5:])
print("Val engine IDs  (all):", val_engine_ids)

train_df = df[df["engine_id"].isin(train_engine_ids)].copy()
val_df   = df[df["engine_id"].isin(val_engine_ids)].copy()

print("\nTrain subset shape:", train_df.shape)
print("Val subset shape  :", val_df.shape)

# ---------- 4. Helper function to build sequences ----------
def build_sequences(data: pd.DataFrame, seq_len: int, feature_cols):

    sequences = []
    labels = []

    # Group by engine so we keep sequences within the same engine
    for eng_id, group in data.groupby("engine_id"):
        group = group.sort_values("cycle")

        feature_values = group[feature_cols].values  # shape: (num_cycles, num_features)
        rul_values = group["RUL"].values            # shape: (num_cycles,)

        num_cycles = len(group)
        if num_cycles < seq_len:
            # if engine has fewer cycles than seq_len, we skip it
            continue

        # Slide a window of length seq_len over this engine's history
        for end_idx in range(seq_len - 1, num_cycles):
            start_idx = end_idx - seq_len + 1
            seq_x = feature_values[start_idx:end_idx + 1]  # (seq_len, num_features)
            seq_y = rul_values[end_idx]                    # RUL at the last cycle

            sequences.append(seq_x)
            labels.append(seq_y)

    X = np.array(sequences)
    y = np.array(labels)

    return X, y


# ---------- 5. Build sequences for train and validation ----------
print("\nBuilding training sequences...")
X_train, y_train = build_sequences(train_df, SEQ_LEN, feature_cols)
print("X_train shape:", X_train.shape)  # (num_train_seq, seq_len, num_features)
print("y_train shape:", y_train.shape)  # (num_train_seq,)

print("\nBuilding validation sequences...")
X_val, y_val = build_sequences(val_df, SEQ_LEN, feature_cols)
print("X_val shape:", X_val.shape)
print("y_val shape:", y_val.shape)

# ---------- 6. Save sequences to .npy files ----------
np.save(X_train_out, X_train)
np.save(y_train_out, y_train)
np.save(X_val_out, X_val)
np.save(y_val_out, y_val)

print(f"\nSaved X_train to: {X_train_out}")
print(f"Saved y_train to: {y_train_out}")
print(f"Saved X_val   to: {X_val_out}")
print(f"Saved y_val   to: {y_val_out}")

print("\nDone.")

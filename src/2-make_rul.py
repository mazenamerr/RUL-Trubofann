import os
import pandas as pd
import matplotlib.pyplot as plt

# ---------- 1. File paths ----------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "Data")

train_path = os.path.join(DATA_DIR, "train_FD003.csv")
output_path = os.path.join(DATA_DIR, "train_FD003_with_RUL.csv")

# ---------- 2. Load training data ----------
train_df = pd.read_csv(train_path)

print("Train shape (before RUL):", train_df.shape)
print("Columns:", train_df.columns.tolist())

# ---------- 3. Compute RUL for each row ----------
max_cycle_per_engine = train_df.groupby("engine_id")["cycle"].transform("max")
train_df["RUL"] = max_cycle_per_engine - train_df["cycle"]

print("\nRUL column added.")
print("Train shape (after RUL):", train_df.shape)

# ---------- 4. Quick sanity check on one engine ----------
example_engine = 1
engine_data = train_df[train_df["engine_id"] == example_engine][["engine_id", "cycle", "RUL"]]

print(f"\nSample rows for engine {example_engine}:")
print(engine_data.head())
print(engine_data.tail())

print("\nOverall RUL stats:")
print("Min RUL:", train_df["RUL"].min())
print("Max RUL:", train_df["RUL"].max())

# ---------- 5. Plot 1: Histogram of all RUL values ----------
plt.figure(figsize=(8, 5))
plt.hist(train_df["RUL"], bins=40, color="steelblue", edgecolor="black")
plt.xlabel("RUL (cycles)")
plt.ylabel("Frequency")
plt.title("Distribution of RUL values (smoothed)")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# ---------- 6. Plot 2: RUL vs cycle for a few engines ----------
example_engines = [1, 10, 50]  #we can change if we want to....I did just for example

plt.figure()
for eng_id in example_engines:
    eng_data = train_df[train_df["engine_id"] == eng_id].sort_values("cycle")
    plt.plot(eng_data["cycle"], eng_data["RUL"], label=f"Engine {eng_id}")

plt.xlabel("Cycle")
plt.ylabel("RUL (cycles)")
plt.title("RUL vs Cycle for selected engines")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()



# ---------- 7. Save the new file ----------
train_df.to_csv(output_path, index=False)
print(f"\nSaved training data with RUL to: {output_path}")

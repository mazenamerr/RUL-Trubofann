import os
import pandas as pd
import matplotlib.pyplot as plt

# ---------- 1. File paths ----------

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "Data")

train_path = os.path.join(DATA_DIR, "train_FD003.csv")
test_path = os.path.join(DATA_DIR, "test_FD003.csv")
rul_path = os.path.join(DATA_DIR, "RUL_FD003.csv")

# ---------- 2. Load data ----------
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
rul_df = pd.read_csv(rul_path)

print("Train shape:", train_df.shape)
print("Test shape :", test_df.shape)
print("RUL shape  :", rul_df.shape)

print("\nColumn names:", train_df.columns.tolist())

# ---------- 3. Basic checks ----------
print("\nNumber of engines in train :", train_df["engine_id"].nunique())
print("Number of engines in test  :", test_df["engine_id"].nunique())

# cycles per engine (train)
cycles_per_unit = train_df.groupby("engine_id")["cycle"].max()
print("\nTrain cycles per engine (first 10):")
print(cycles_per_unit.head(10))

print("\nMin cycles:", cycles_per_unit.min())
print("Max cycles:", cycles_per_unit.max())

# ---------- 4. Simple plot ----------

example_unit = 1
example_sensor = "s3"

unit_data = train_df[train_df["engine_id"] == example_unit]

plt.figure()
plt.plot(unit_data["cycle"], unit_data[example_sensor])
plt.xlabel("Cycle")
plt.ylabel(example_sensor)
plt.title(f"Engine {example_unit} - {example_sensor} over time")
plt.grid(True)
plt.tight_layout()
plt.show()

import sys
import os

sys.path.append(os.path.abspath("src"))

from load_data import load_single_dataset
from preprocess import add_rul, clean_and_scale
from sequence import create_sequences
from baseline_model import train_and_save

os.makedirs("model", exist_ok=True)

datasets = [
    ("FD001", "data/train_FD001.txt"),
    ("FD002", "data/train_FD002.txt"),
    ("FD003", "data/train_FD003.txt"),
    ("FD004", "data/train_FD004.txt")
]

for name, path in datasets:
    print(f"\n🔹 Training model for {name}")

    df = load_single_dataset(path)
    df = add_rul(df)
    df, sensor_cols = clean_and_scale(df)

    X, y = create_sequences(df, sensor_cols, window_size=20)

    # Laptop-safe subset
    X = X[:20000]
    y = y[:20000]

    model_path = f"model/rul_model_{name.lower()}.pkl"
    rmse = train_and_save(X, y, model_path)

    print(f"✅ {name} training completed | RMSE: {rmse}")
    print(f"📦 Model saved at: {model_path}")

print("\n🎉 All datasets trained one by one successfully!")
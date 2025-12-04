import argparse, os, pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def preprocess_household(input_path, out_dir, seq_len=168, stride=24):
    os.makedirs(out_dir, exist_ok=True)

    print("Reading raw data...")
    df = pd.read_csv(
        input_path,
        sep=";",
        na_values="?",
        low_memory=False
    )

    # Combine date + time
    df["Datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"],
                                    format="%d/%m/%Y %H:%M:%S",
                                    errors="coerce")
    df = df.drop(columns=["Date", "Time"])
    df = df.set_index("Datetime")
    df = df.sort_index()

    # Convert all columns to numeric
    df = df.apply(pd.to_numeric, errors="coerce")

    print("Initial shape:", df.shape)
    print("Missing values per column:\n", df.isna().sum().head())

    # Resample to hourly mean
    df = df.resample("H").mean()

    # Interpolate small gaps
    df = df.interpolate(limit_direction="both")

    # Optionally drop rows still NaN after interpolation
    df = df.dropna()
    print("After resampling and interpolation:", df.shape)

    # Convert to numpy
    data = df.values.astype(np.float32)
    feature_names = df.columns.tolist()
    print("Features:", feature_names)

    # --- Windowing ---
    def make_windows(arr, seq_len, stride):
        N, D = arr.shape
        windows = []
        for start in range(0, N - seq_len + 1, stride):
            windows.append(arr[start:start + seq_len])
        return np.stack(windows)

    windows = make_windows(data, seq_len, stride)
    print("Windowed shape:", windows.shape)  # (num_windows, 168, D)

    # --- Train/Val/Test split ---
    n = windows.shape[0]
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)
    train, val, test = (
        windows[:n_train],
        windows[n_train:n_train + n_val],
        windows[n_train + n_val:],
    )

    # --- Scaling ---
    scaler = StandardScaler()
    D = train.shape[2]
    scaler.fit(train.reshape(-1, D))
    def scale(x):
        return scaler.transform(x.reshape(-1, D)).reshape(x.shape)

    train_s, val_s, test_s = scale(train), scale(val), scale(test)

    # --- Save ---
    np.save(os.path.join(out_dir, "train.npy"), train_s)
    np.save(os.path.join(out_dir, "val.npy"), val_s)
    np.save(os.path.join(out_dir, "test.npy"), test_s)
    with open(os.path.join(out_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    with open(os.path.join(out_dir, "features.txt"), "w") as f:
        f.write("\n".join(feature_names))

    print("Saved arrays to", out_dir)
    print("Train/Val/Test shapes:", train_s.shape, val_s.shape, test_s.shape)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True,
                   help="Path to household_power_consumption.txt")
    p.add_argument("--out_dir", default="data/processed/electricity")
    p.add_argument("--seq_len", type=int, default=168)
    p.add_argument("--stride", type=int, default=24)
    args = p.parse_args()
    preprocess_household(args.input, args.out_dir, args.seq_len, args.stride)

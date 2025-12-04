import argparse, os, pickle, json
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def preprocess_household(input_path, out_dir, seq_len=168, stride=12, smooth=True):
    os.makedirs(out_dir, exist_ok=True)

    print("Reading raw data...")
    df = pd.read_csv(
        input_path,
        sep=";",
        na_values="?",
        low_memory=False
    )

    # --- Parse datetime ---
    df["Datetime"] = pd.to_datetime(
        df["Date"] + " " + df["Time"],
        format="%d/%m/%Y %H:%M:%S",
        errors="coerce"
    )
    df = df.drop(columns=["Date", "Time"])
    df = df.set_index("Datetime").sort_index()

    # Convert all columns to numeric
    df = df.apply(pd.to_numeric, errors="coerce")
    print("Initial shape:", df.shape)

    # --- Hourly resample and interpolate ---
    df = df.resample("H").mean()
    df = df.interpolate(limit_direction="both")

    if smooth:
        # Light rolling mean to smooth spikes
        df = df.rolling(window=3, min_periods=1, center=True).mean()

    df = df.dropna()
    print("After resampling/interpolation:", df.shape)

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
    print("Windowed shape:", windows.shape)

    # --- Train/Val/Test split (reproducible shuffle) ---
    np.random.seed(42)
    np.random.shuffle(windows)

    n = windows.shape[0]
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)
    train, val, test = (
        windows[:n_train],
        windows[n_train:n_train + n_val],
        windows[n_train + n_val:]
    )

    # --- Per-feature scaling ---
    D = train.shape[2]
    scalers = []
    train_s = np.empty_like(train)
    val_s = np.empty_like(val)
    test_s = np.empty_like(test)

    for d in range(D):
        s = StandardScaler()
        s.fit(train[:, :, d].reshape(-1, 1))
        scalers.append(s)

        train_s[:, :, d] = s.transform(train[:, :, d].reshape(-1, 1)).reshape(train[:, :, d].shape)
        val_s[:, :, d] = s.transform(val[:, :, d].reshape(-1, 1)).reshape(val_s[:, :, d].shape)
        test_s[:, :, d] = s.transform(test[:, :, d].reshape(-1, 1)).reshape(test_s[:, :, d].shape)

    # --- Save processed arrays and scalers ---
    np.save(os.path.join(out_dir, "train.npy"), train_s)
    np.save(os.path.join(out_dir, "val.npy"), val_s)
    np.save(os.path.join(out_dir, "test.npy"), test_s)

    with open(os.path.join(out_dir, "scalers.pkl"), "wb") as f:
        pickle.dump(scalers, f)

    with open(os.path.join(out_dir, "features.txt"), "w") as f:
        f.write("\n".join(feature_names))

    meta = {
        "seq_len": seq_len,
        "stride": stride,
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n - n_train - n_val,
        "features": feature_names,
        "smooth": smooth
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("Saved arrays and metadata to", out_dir)
    print("Train/Val/Test shapes:", train_s.shape, val_s.shape, test_s.shape)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True,
                   help="Path to household_power_consumption.txt")
    p.add_argument("--out_dir", default="data/processed/electricity")
    p.add_argument("--seq_len", type=int, default=168)
    p.add_argument("--stride", type=int, default=12)
    p.add_argument("--no_smooth", action="store_true",
                   help="Disable rolling mean smoothing")
    args = p.parse_args()

    preprocess_household(
        args.input,
        args.out_dir,
        args.seq_len,
        args.stride,
        smooth=not args.no_smooth
    )

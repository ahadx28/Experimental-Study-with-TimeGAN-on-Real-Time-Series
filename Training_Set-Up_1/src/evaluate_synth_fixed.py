# src/evaluate_synth_fixed.py
import os, sys, math, random, json, pickle
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics import mean_squared_error
from fastdtw import fastdtw
from sklearn.linear_model import Ridge
import warnings
warnings.filterwarnings("ignore")

# --- paths ---
DATA_DIR = "data/processed/electricity"
SYN_DIR = "outputs/synth"
OUT_DIR = "outputs/eval"
os.makedirs(OUT_DIR, exist_ok=True)

# --- load data ---
def load_real_scaled():
    real_test = np.load(os.path.join(DATA_DIR, "test.npy"))   # scaled (by scaler fitted on train)
    return real_test

def load_synth_scaled():
    # prefer files named synth_electricity_*.npy; otherwise pick the first .npy
    files = sorted([f for f in os.listdir(SYN_DIR) if f.endswith(".npy")])
    if not files:
        raise SystemExit("No .npy files in outputs/synth/")
    # pick the largest synth file (most windows) — usually good
    files_sorted = sorted(files, key=lambda f: os.path.getsize(os.path.join(SYN_DIR, f)), reverse=True)
    synth_path = os.path.join(SYN_DIR, files_sorted[0])
    print("Loading synth file:", synth_path)
    synth_inv = np.load(synth_path)          # this is inverse-scaled (we saved inverse-scaled earlier)
    # convert inverse-scaled synth back to scaled using scaler fitted on train
    with open(os.path.join(DATA_DIR, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    n, T, D = synth_inv.shape
    synth_scaled = scaler.transform(synth_inv.reshape(-1, D)).reshape(n, T, D)
    return synth_scaled

# --- Metrics ---
def mmd_rbf(X, Y, sigma=None):
    Xf = X.reshape(X.shape[0], -1)
    Yf = Y.reshape(Y.shape[0], -1)
    if sigma is None:
        d = cdist(Xf, Xf, 'euclidean')
        med = np.median(d[d>0]) if np.any(d>0) else 1.0
        sigma = med if med>0 else 1.0
    Kxx = np.exp(-cdist(Xf, Xf, 'sqeuclidean')/(2*sigma**2))
    Kyy = np.exp(-cdist(Yf, Yf, 'sqeuclidean')/(2*sigma**2))
    Kxy = np.exp(-cdist(Xf, Yf, 'sqeuclidean')/(2*sigma**2))
    return float(Kxx.mean() + Kyy.mean() - 2*Kxy.mean()), float(sigma)

def avg_dtw(X, Y, n_pairs=200, agg='sum'):
    Nx, Ny = X.shape[0], Y.shape[0]
    s = 0.0
    for _ in range(n_pairs):
        i = random.randrange(Nx)
        j = random.randrange(Ny)
        xi = X[i]
        yj = Y[j]
        if agg == 'sum':
            xi_agg = xi.sum(axis=1)
            yj_agg = yj.sum(axis=1)
        elif agg == 'first':
            xi_agg = xi[:,0]
            yj_agg = yj[:,0]
        else:
            xi_agg = xi.mean(axis=1)
            yj_agg = yj.mean(axis=1)
        dist, _ = fastdtw(xi_agg, yj_agg)
        s += dist
    return s / n_pairs

def predictive_utility(real_train, synth_train, real_test, n_epochs=40):
    # Flatten first half of windows -> predict mean of second half's first feature.
    def prepare(X):
        inputs = X[:, :X.shape[1]//2, :].reshape(X.shape[0], -1)
        targets = X[:, X.shape[1]//2:, 0].mean(axis=1)
        return inputs, targets
    Xr, Yr = prepare(real_train)
    Xs, Ys = prepare(synth_train)
    Xt, Yt = prepare(real_test)
    # Fit ridge (fast, stable)
    r1 = Ridge(alpha=1.0)
    r1.fit(Xr, Yr)
    pred_real = r1.predict(Xt)
    mse_real = mean_squared_error(Yt, pred_real)
    r2 = Ridge(alpha=1.0)
    r2.fit(Xs, Ys)
    pred_synth = r2.predict(Xt)
    mse_synth = mean_squared_error(Yt, pred_synth)
    return float(mse_real), float(mse_synth)

# --- run evaluation ---
def run():
    real_test = load_real_scaled()
    synth = load_synth_scaled()

    # sample subsets to keep compute reasonable
    n_sample = min(500, real_test.shape[0], synth.shape[0])
    real_samp = real_test[:n_sample]
    synth_samp = synth[:n_sample]
    print("Using", n_sample, "windows for metrics (scaled)")

    out = {}
    # MMD
    mmd_val, sigma = mmd_rbf(real_samp, synth_samp, sigma=None)
    out['mmd_rbf'] = mmd_val
    out['mmd_sigma'] = sigma
    print(f"MMD (RBF) = {mmd_val:.6g}  (sigma={sigma:.6g})")

    # DTW (aggregated)
    dtw_val = avg_dtw(real_samp, synth_samp, n_pairs=200, agg='sum')
    out['avg_dtw'] = dtw_val
    print(f"Avg DTW (sum-agg) over 200 random pairs = {dtw_val:.4f}")

    # Predictive utility
    real_train = np.load(os.path.join(DATA_DIR, "train.npy"))
    # take subsets for speed
    rt = real_train[:800] if real_train.shape[0] > 800 else real_train
    st = synth[:800] if synth.shape[0] > 800 else synth
    mse_real, mse_synth = predictive_utility(rt, st, real_test[:200])
    out['mse_predict_real_trained'] = mse_real
    out['mse_predict_synth_trained'] = mse_synth
    print(f"Predictive MSE (train on real -> test on real) = {mse_real:.6f}")
    print(f"Predictive MSE (train on synth -> test on real) = {mse_synth:.6f}")
    out['predictive_mse_ratio'] = mse_synth / (mse_real + 1e-12)

    # Save JSON report
    report_path = os.path.join(OUT_DIR, "eval_report.json")
    with open(report_path, "w") as f:
        json.dump(out, f, indent=2)
    print("Saved JSON report to", report_path)

    # Save a simple CSV with numeric summary
    import csv
    csv_path = os.path.join(OUT_DIR, "eval_summary.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        for k,v in out.items():
            w.writerow([k, v])
    print("Saved CSV summary to", csv_path)
    return out

if __name__ == "__main__":
    res = run()
    print("Done. Results:", res)

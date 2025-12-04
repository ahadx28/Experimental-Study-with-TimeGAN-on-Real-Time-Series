# src/generate_and_save.py (updated for WGAN-GP training)
import os
import pickle
import numpy as np
import tensorflow as tf
from timegan_tf import Embedder, Recovery, Generator, Supervisor, Discriminator

# ----------------- Paths -----------------
DATA_DIR = "data/processed/electricity"
CKPT_DIR = "outputs/checkpoints/timegan_wgan_gp"   # match new WGAN-GP directory
OUT_DIR = "outputs/synth"
os.makedirs(OUT_DIR, exist_ok=True)

# ----------------- Model config (match training) -----------------
SEQ_LEN = 168
FEATURE_DIM = np.load(os.path.join(DATA_DIR, "train.npy")).shape[2]
HIDDEN_DIM = 64
Z_DIM = 16     # must match new training config!

# ----------------- Rebuild models -----------------
embedder = Embedder(input_dim=FEATURE_DIM, hidden_dim=HIDDEN_DIM, num_layers=2)
recovery = Recovery(hidden_dim=HIDDEN_DIM, output_dim=FEATURE_DIM, num_layers=2)
generator = Generator(z_dim=Z_DIM, hidden_dim=HIDDEN_DIM, num_layers=2)
supervisor = Supervisor(hidden_dim=HIDDEN_DIM, num_layers=1)
discriminator = Discriminator(hidden_dim=HIDDEN_DIM, num_layers=1)

# build (dummy)
_ = embedder(tf.zeros([1, SEQ_LEN, FEATURE_DIM]))
_ = recovery(tf.zeros([1, SEQ_LEN, HIDDEN_DIM]))
_ = generator(tf.zeros([1, SEQ_LEN, Z_DIM]))
_ = supervisor(tf.zeros([1, SEQ_LEN, HIDDEN_DIM]))
_ = discriminator(tf.zeros([1, SEQ_LEN, HIDDEN_DIM]))

# ----------------- Restore latest checkpoint -----------------
ckpt = tf.train.Checkpoint(embedder=embedder, recovery=recovery,
                           generator=generator, supervisor=supervisor,
                           discriminator=discriminator)
manager = tf.train.CheckpointManager(ckpt, CKPT_DIR, max_to_keep=6)
if manager.latest_checkpoint:
    ckpt.restore(manager.latest_checkpoint).expect_partial()
    print("Restored checkpoint:", manager.latest_checkpoint)
else:
    raise SystemExit(f"No checkpoint found in {CKPT_DIR}")

# ----------------- Load scaler(s) -----------------
scalers_path = os.path.join(DATA_DIR, "scalers.pkl")
if not os.path.exists(scalers_path):
    scalers_path = os.path.join(DATA_DIR, "scaler.pkl")
if not os.path.exists(scalers_path):
    raise SystemExit("No scaler file found in data directory.")
with open(scalers_path, "rb") as f:
    scalers = pickle.load(f)

# ----------------- Generate synthetic windows -----------------
def generate(n_samples, batch_size=64, inverse_scale=True):
    """Generate synthetic sequences and inverse scale them to real units."""
    out = []
    for i in range(0, n_samples, batch_size):
        b = min(batch_size, n_samples - i)
        Z = tf.random.normal([b, SEQ_LEN, Z_DIM])
        Ehat = generator(Z, training=False)
        Hhat = supervisor(Ehat, training=False)
        Xhat = recovery(Hhat, training=False).numpy()  # scaled outputs
        out.append(Xhat)
    X = np.concatenate(out, axis=0)

    if not inverse_scale:
        return X

    n, T, D = X.shape
    X_inv = np.empty_like(X)
    if isinstance(scalers, list):
        # Per-feature scaling
        for d, s in enumerate(scalers):
            X_inv[:, :, d] = s.inverse_transform(X[:, :, d].reshape(-1, 1)).reshape(n, T)
    else:
        # Single scaler fallback
        X_inv = scalers.inverse_transform(X.reshape(-1, D)).reshape(n, T, D)
    return X_inv

# ----------------- Run -----------------
if __name__ == "__main__":
    N = 2000   # number of windows to generate
    synth = generate(N, batch_size=64, inverse_scale=True)
    out_npy = os.path.join(OUT_DIR, f"synth_electricity_{N}w.npy")
    out_csv = os.path.join(OUT_DIR, f"synth_electricity_{N}w.csv")
    np.save(out_npy, synth)
    np.savetxt(out_csv, synth.reshape(synth.shape[0], -1), delimiter=",")
    print(f"Saved {synth.shape} synthetic windows to:")
    print(" ", out_npy)
    print(" ", out_csv)

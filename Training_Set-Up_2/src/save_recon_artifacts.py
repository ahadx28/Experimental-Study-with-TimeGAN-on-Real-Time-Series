# src/save_recon_artifacts.py
import os, pickle
import numpy as np
import tensorflow as tf
from timegan_tf import Embedder, Recovery

# ---------------- Configuration ----------------
DATA_DIR = "data/processed/electricity"
CKPT_DIR = "outputs/checkpoints/recon"

SEQ_LEN = 168
HIDDEN_DIM = 64

# ---------------- Model Reconstruction ----------------
feature_dim = np.load(os.path.join(DATA_DIR, "train.npy")).shape[2]
embedder = Embedder(input_dim=feature_dim, hidden_dim=HIDDEN_DIM, num_layers=2)
recovery = Recovery(hidden_dim=HIDDEN_DIM, output_dim=feature_dim, num_layers=2)

# Build model shapes
_ = embedder(tf.zeros([1, SEQ_LEN, feature_dim]))
_ = recovery(tf.zeros([1, SEQ_LEN, HIDDEN_DIM]))

# ---------------- Checkpoint Loading ----------------
ckpt = tf.train.Checkpoint(embedder=embedder, recovery=recovery)
manager = tf.train.CheckpointManager(ckpt, CKPT_DIR, max_to_keep=5)

if manager.latest_checkpoint:
    ckpt.restore(manager.latest_checkpoint).expect_partial()
    print("Restored from checkpoint:", manager.latest_checkpoint)
else:
    print("⚠️ No checkpoint found in", CKPT_DIR)
    raise SystemExit("Please re-run train_recon_tf.py before saving artifacts.")

# ---------------- Save final .weights.h5 ----------------
embedder_path = os.path.join(CKPT_DIR, "embedder_final.weights.h5")
recovery_path = os.path.join(CKPT_DIR, "recovery_final.weights.h5")
embedder.save_weights(embedder_path)
recovery.save_weights(recovery_path)
print("✅ Saved weights:", embedder_path, "and", recovery_path)

# ---------------- Reconstruction Example Generation ----------------
test = np.load(os.path.join(DATA_DIR, "test.npy"))
examples = test[:3]

# Forward pass
h_ex = embedder(examples, training=False)
x_rec = recovery(h_ex, training=False).numpy()

# Load scaler (support both single and per-feature)
scalers_path = os.path.join(DATA_DIR, "scalers.pkl")
if not os.path.exists(scalers_path):
    scalers_path = os.path.join(DATA_DIR, "scaler.pkl")

with open(scalers_path, "rb") as f:
    scaler = pickle.load(f)

# Inverse scaling
n, T, D = x_rec.shape
if isinstance(scaler, list):
    # per-feature scalers
    x_rec_inv = np.zeros_like(x_rec)
    x_orig_inv = np.zeros_like(examples)
    for j, sc in enumerate(scaler):
        x_rec_inv[..., j] = sc.inverse_transform(x_rec[..., j])
        x_orig_inv[..., j] = sc.inverse_transform(examples[..., j])
else:
    # single scaler
    x_rec_inv = scaler.inverse_transform(x_rec.reshape(-1, D)).reshape(n, T, D)
    x_orig_inv = scaler.inverse_transform(examples.reshape(-1, D)).reshape(n, T, D)

# Save reconstruction results
np.save(os.path.join(CKPT_DIR, "recon_examples_orig.npy"), x_orig_inv)
np.save(os.path.join(CKPT_DIR, "recon_examples_rec.npy"), x_rec_inv)

# ---------------- Diagnostics ----------------
mae = float(np.mean(np.abs(x_orig_inv - x_rec_inv)))
print("✅ Reconstruction complete.")
print("Reconstruction examples saved to:", CKPT_DIR)
print(f"Mean Absolute Error (first feature, all samples): {mae:.6f}")

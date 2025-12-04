# src/save_recon_artifacts.py
import os, pickle
import numpy as np
import tensorflow as tf
from timegan_tf import Embedder, Recovery

DATA_DIR = "data/processed/electricity"
CKPT_DIR = "outputs/checkpoints/recon"

SEQ_LEN = 168
HIDDEN_DIM = 64

# Rebuild models (must match training config: num_layers etc)
feature_dim = np.load(os.path.join(DATA_DIR, "train.npy")).shape[2]
embedder = Embedder(input_dim=feature_dim, hidden_dim=HIDDEN_DIM, num_layers=2)
recovery = Recovery(hidden_dim=HIDDEN_DIM, output_dim=feature_dim, num_layers=2)

# Build shapes
_ = embedder(tf.zeros([1, SEQ_LEN, feature_dim]))
_ = recovery(tf.zeros([1, SEQ_LEN, HIDDEN_DIM]))

# Checkpoint manager (should point to same dir used in training)
ckpt = tf.train.Checkpoint(embedder=embedder, recovery=recovery)
manager = tf.train.CheckpointManager(ckpt, CKPT_DIR, max_to_keep=5)

if manager.latest_checkpoint:
    ckpt.restore(manager.latest_checkpoint).expect_partial()
    print("Restored from:", manager.latest_checkpoint)
else:
    print("No TF checkpoint found in", CKPT_DIR)
    # still proceed to save current random weights if you want
    # but better to re-run training if no checkpoint exists
    # exit
    raise SystemExit("No checkpoint to restore. Re-run training or point CKPT_DIR correctly.")

# Save correct .weights.h5 files
embedder_path = os.path.join(CKPT_DIR, "embedder_final.weights.h5")
recovery_path = os.path.join(CKPT_DIR, "recovery_final.weights.h5")
embedder.save_weights(embedder_path)
recovery.save_weights(recovery_path)
print("Saved weights:", embedder_path, recovery_path)

# Generate reconstruction examples (inverse-scale) from test set
test = np.load(os.path.join(DATA_DIR, "test.npy"))
examples = test[:3]
h_ex = embedder(examples, training=False)
x_rec = recovery(h_ex, training=False).numpy()

with open(os.path.join(DATA_DIR, "scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)

n, T, D = x_rec.shape
x_rec_inv = scaler.inverse_transform(x_rec.reshape(-1, D)).reshape(n, T, D)
x_orig_inv = scaler.inverse_transform(examples.reshape(-1, D)).reshape(n, T, D)

np.save(os.path.join(CKPT_DIR, "recon_examples_orig.npy"), x_orig_inv)
np.save(os.path.join(CKPT_DIR, "recon_examples_rec.npy"), x_rec_inv)
print("Saved reconstruction examples to", CKPT_DIR)
print("Mean abs error (first example, first feature):",
      float(np.mean(np.abs(x_orig_inv[0,:,0] - x_rec_inv[0,:,0]))))

import os, pickle, json
import numpy as np
import tensorflow as tf
from tensorflow.keras import optimizers
from timegan_tf import Embedder, Recovery

# Config
DATA_DIR = "data/processed/electricity"
CKPT_DIR = "outputs/checkpoints/recon"
os.makedirs(CKPT_DIR, exist_ok=True)

SEQ_LEN = 168
BATCH_SIZE = 32
HIDDEN_DIM = 64
NUM_EPOCHS = 300
LR = 5e-4
PATIENCE = 10

# Load data
train = np.load(os.path.join(DATA_DIR, "train.npy"))
val = np.load(os.path.join(DATA_DIR, "val.npy"))
test = np.load(os.path.join(DATA_DIR, "test.npy"))
print("Shapes: train", train.shape, "val", val.shape, "test", test.shape)

# Build tf.data
train_ds = tf.data.Dataset.from_tensor_slices(train).shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
val_ds = tf.data.Dataset.from_tensor_slices(val).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# Model
feature_dim = train.shape[2]
embedder = Embedder(input_dim=feature_dim, hidden_dim=HIDDEN_DIM, num_layers=2, dropout=0.1)
recovery = Recovery(hidden_dim=HIDDEN_DIM, output_dim=feature_dim, num_layers=2, dropout=0.1)

# Build once
_ = embedder(tf.zeros([1, SEQ_LEN, feature_dim]))
_ = recovery(tf.zeros([1, SEQ_LEN, HIDDEN_DIM]))

# Optimizer + loss
optimizer = optimizers.Adam(learning_rate=LR)
mse = tf.keras.losses.MeanSquaredError()

# Checkpoint manager
ckpt = tf.train.Checkpoint(optimizer=optimizer, embedder=embedder, recovery=recovery)
manager = tf.train.CheckpointManager(ckpt, CKPT_DIR, max_to_keep=3)

# TensorBoard logging
train_writer = tf.summary.create_file_writer(os.path.join(CKPT_DIR, "logs"))

@tf.function
def train_step(x):
    with tf.GradientTape() as tape:
        h = embedder(x, training=True)
        x_tilde = recovery(h, training=True)
        loss = mse(x, x_tilde)
    vars_ = embedder.trainable_variables + recovery.trainable_variables
    grads = tape.gradient(loss, vars_)
    optimizer.apply_gradients(zip(grads, vars_))
    return loss

@tf.function
def val_step(x):
    h = embedder(x, training=False)
    x_tilde = recovery(h, training=False)
    return mse(x, x_tilde)

best_val = 1e9
wait = 0
for epoch in range(1, NUM_EPOCHS + 1):
    train_losses = [train_step(b).numpy() for b in train_ds]
    val_losses = [val_step(b).numpy() for b in val_ds]

    train_loss = float(np.mean(train_losses))
    val_loss = float(np.mean(val_losses))

    with train_writer.as_default():
        tf.summary.scalar("train_loss", train_loss, step=epoch)
        tf.summary.scalar("val_loss", val_loss, step=epoch)

    print(f"[Recon] Epoch {epoch:03d}: train={train_loss:.6f}, val={val_loss:.6f}")

    if val_loss < best_val:
        best_val = val_loss
        wait = 0
        manager.save()
        print(f"  -> checkpoint saved (val {best_val:.6f})")
    else:
        wait += 1
        if wait >= PATIENCE:
            print(f"Early stopping at epoch {epoch}")
            break

# Save final weights
embedder.save_weights(os.path.join(CKPT_DIR, "embedder_final.h5"))
recovery.save_weights(os.path.join(CKPT_DIR, "recovery_final.h5"))

# Save reconstruction examples (inverse scaled)
with open(os.path.join(DATA_DIR, "scalers.pkl"), "rb") as f:
    scalers = pickle.load(f)

examples = test[:3]  # (3, T, D)
h_ex = embedder(examples, training=False)
x_rec = recovery(h_ex, training=False).numpy()

n, T, D = x_rec.shape
x_rec_inv = np.empty_like(x_rec)
x_orig_inv = np.empty_like(examples)

for d, s in enumerate(scalers):
    x_rec_inv[:, :, d] = s.inverse_transform(x_rec[:, :, d].reshape(-1, 1)).reshape(x_rec_inv[:, :, d].shape)
    x_orig_inv[:, :, d] = s.inverse_transform(examples[:, :, d].reshape(-1, 1)).reshape(x_orig_inv[:, :, d].shape)

np.save(os.path.join(CKPT_DIR, "recon_examples_orig.npy"), x_orig_inv)
np.save(os.path.join(CKPT_DIR, "recon_examples_rec.npy"), x_rec_inv)
print("Saved recon examples to", CKPT_DIR)
print("Mean abs error (first example, first feature):", np.mean(np.abs(x_orig_inv[0,:,0] - x_rec_inv[0,:,0])))

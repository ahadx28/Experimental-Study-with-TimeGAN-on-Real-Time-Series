# src/generate_and_save.py
import os, pickle, numpy as np, tensorflow as tf
from timegan_tf import Embedder, Recovery, Generator, Supervisor, Discriminator

DATA_DIR = "data/processed/electricity"
CKPT_DIR = "outputs/checkpoints/timegan_adv"
OUT_DIR = "outputs/synth"
os.makedirs(OUT_DIR, exist_ok=True)

SEQ_LEN = 168
FEATURE_DIM = np.load(os.path.join(DATA_DIR, "train.npy")).shape[2]
HIDDEN_DIM = 64
Z_DIM = 32

# rebuild models (match training config)
embedder = Embedder(input_dim=FEATURE_DIM, hidden_dim=HIDDEN_DIM, num_layers=2)
recovery = Recovery(hidden_dim=HIDDEN_DIM, output_dim=FEATURE_DIM, num_layers=2)
generator = Generator(z_dim=Z_DIM, hidden_dim=HIDDEN_DIM, num_layers=2)
supervisor = Supervisor(hidden_dim=HIDDEN_DIM, num_layers=1)
discriminator = Discriminator(hidden_dim=HIDDEN_DIM, num_layers=1)

# build
_ = embedder(tf.zeros([1, SEQ_LEN, FEATURE_DIM]))
_ = recovery(tf.zeros([1, SEQ_LEN, HIDDEN_DIM]))
_ = generator(tf.zeros([1, SEQ_LEN, Z_DIM]))
_ = supervisor(tf.zeros([1, SEQ_LEN, HIDDEN_DIM]))
_ = discriminator(tf.zeros([1, SEQ_LEN, HIDDEN_DIM]))

# restore latest adv checkpoint
ckpt = tf.train.Checkpoint(embedder=embedder, recovery=recovery,
                           generator=generator, supervisor=supervisor,
                           discriminator=discriminator)
manager = tf.train.CheckpointManager(ckpt, CKPT_DIR, max_to_keep=5)
if manager.latest_checkpoint:
    ckpt.restore(manager.latest_checkpoint).expect_partial()
    print("Restored:", manager.latest_checkpoint)
else:
    raise SystemExit("No checkpoint found in "+CKPT_DIR)

# load scaler
with open(os.path.join(DATA_DIR, "scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)

def generate(n_samples, batch_size=64):
    out = []
    for i in range(0, n_samples, batch_size):
        b = min(batch_size, n_samples - i)
        Z = tf.random.normal([b, SEQ_LEN, Z_DIM])
        Ehat = generator(Z, training=False)
        Hhat = supervisor(Ehat, training=False)
        Xhat = recovery(Hhat, training=False).numpy()  # scaled
        out.append(Xhat)
    X = np.concatenate(out, axis=0)
    # inverse scale
    n, T, D = X.shape
    X_inv = scaler.inverse_transform(X.reshape(-1, D)).reshape(n, T, D)
    return X_inv

if __name__ == "__main__":
    N = 2000   # how many windows to generate
    synth = generate(N, batch_size=64)
    np.save(os.path.join(OUT_DIR, "synth_electricity_{}w.npy".format(N)), synth)
    # also save CSV (flattened per-window -> rows)
    s_flat = synth.reshape(synth.shape[0], -1)
    np.savetxt(os.path.join(OUT_DIR, "synth_electricity_{}w.csv".format(N)), s_flat, delimiter=",")
    print(f"Saved {synth.shape} synthetic windows to {OUT_DIR}")

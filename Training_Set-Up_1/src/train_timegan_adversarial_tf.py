# src/train_timegan_adversarial_tf.py
import os
import time
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras import optimizers
from timegan_tf import Embedder, Recovery, Generator, Supervisor, Discriminator

# ----------------- Config -----------------
DATA_DIR = "data/processed/electricity"
CKPT_DIR = "outputs/checkpoints/timegan_adv"
os.makedirs(CKPT_DIR, exist_ok=True)

SEQ_LEN = 168
FEATURE_DIM = np.load(os.path.join(DATA_DIR, "train.npy")).shape[2]
HIDDEN_DIM = 64
Z_DIM = 32              # noise dim per time step
BATCH_SIZE = 32
LR = 1e-4

# training lengths
EPOCHS_SUPERVISOR = 60      # supervised pretrain
EPOCHS_ADVERSARIAL = 300    # adversarial outer epochs (you can increase)
STEPS_PER_EPOCH = None      # None => iterate over dataset once per epoch

# loss weights (tune these)
LAMBDA_SUP = 100.0
LAMBDA_REC = 10.0
LAMBDA_MOMENT = 100.0  # optional

# ----------------- Load data -----------------
train = np.load(os.path.join(DATA_DIR, "train.npy"))
val = np.load(os.path.join(DATA_DIR, "val.npy"))
test = np.load(os.path.join(DATA_DIR, "test.npy"))
print("Loaded shapes:", train.shape, val.shape, test.shape)

train_ds = tf.data.Dataset.from_tensor_slices(train).shuffle(2000).batch(BATCH_SIZE, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
val_ds = tf.data.Dataset.from_tensor_slices(val).batch(BATCH_SIZE, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

# ----------------- Build models -----------------
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

# ----------------- Optimizers & losses -----------------
opt_e = optimizers.Adam(LR)
opt_r = optimizers.Adam(LR)
opt_g = optimizers.Adam(LR)
opt_s = optimizers.Adam(LR)
opt_d = optimizers.Adam(LR)

mse = tf.keras.losses.MeanSquaredError()
bce_logits = tf.keras.losses.BinaryCrossentropy(from_logits=True)

# checkpoint manager
ckpt = tf.train.Checkpoint(embedder=embedder, recovery=recovery,
                           generator=generator, supervisor=supervisor,
                           discriminator=discriminator,
                           opt_e=opt_e, opt_r=opt_r, opt_g=opt_g, opt_s=opt_s, opt_d=opt_d)
manager = tf.train.CheckpointManager(ckpt, CKPT_DIR, max_to_keep=5)

# Optionally restore recon-trained weights (if available)
recon_ckpt_dir = "outputs/checkpoints/recon"
# if you used CheckpointManager earlier, simply restore its latest
if os.path.isdir(recon_ckpt_dir):
    try:
        # attempt to restore embedder+recovery from recon manager
        recon_ckpt = tf.train.Checkpoint(embedder=embedder, recovery=recovery)
        recon_manager = tf.train.CheckpointManager(recon_ckpt, recon_ckpt_dir, max_to_keep=3)
        if recon_manager.latest_checkpoint:
            recon_ckpt.restore(recon_manager.latest_checkpoint).expect_partial()
            print("Restored recon checkpoint:", recon_manager.latest_checkpoint)
    except Exception as e:
        print("No recon checkpoint restored:", e)

# ----------------- Helpers -----------------
def sample_z(batch_size):
    return tf.random.normal([batch_size, SEQ_LEN, Z_DIM])

def rbf_kernel(x, y, sigma):
    # x: (n,d), y:(m,d)
    xx = tf.reduce_sum(tf.square(x), axis=1, keepdims=True)
    yy = tf.reduce_sum(tf.square(y), axis=1, keepdims=True)
    xy = tf.matmul(x, y, transpose_b=True)
    d2 = xx - 2.0*xy + tf.transpose(yy)
    return tf.exp(-d2 / (2.0 * sigma * sigma))

def mmd_rbf(X, Y, sigma=None):
    # flatten sequences into vectors
    Xf = tf.reshape(X, [tf.shape(X)[0], -1])
    Yf = tf.reshape(Y, [tf.shape(Y)[0], -1])
    if sigma is None:
        # median heuristic on sample
        D = tf.norm(tf.expand_dims(Xf,1)-tf.expand_dims(Xf,0), axis=-1)
        med = tf.experimental.numpy.median(tf.boolean_mask(D, D>0))
        sigma = tf.maximum(med, 1.0)
    Kxx = rbf_kernel(Xf, Xf, sigma)
    Kyy = rbf_kernel(Yf, Yf, sigma)
    Kxy = rbf_kernel(Xf, Yf, sigma)
    return tf.reduce_mean(Kxx) + tf.reduce_mean(Kyy) - 2.0 * tf.reduce_mean(Kxy)

# ----------------- Training steps -----------------
@tf.function
def supervised_step(x):
    # Train supervisor to predict next step in latent space (using embedder's H)
    with tf.GradientTape() as tape:
        H = embedder(x, training=False)   # keep embedder frozen (paper trains sup using H from embedder)
        H_hat = supervisor(H, training=True)
        # supervised loss compares shifted H
        loss_s = mse(H[:,1:,:], H_hat[:, :-1, :])
    grads = tape.gradient(loss_s, supervisor.trainable_variables)
    opt_s.apply_gradients(zip(grads, supervisor.trainable_variables))
    return loss_s

@tf.function
def discriminator_step(x):
    batch_size = tf.shape(x)[0]
    z = sample_z(batch_size)
    with tf.GradientTape() as tape:
        # real latent from embedder
        H_real = embedder(x, training=False)
        D_real = discriminator(H_real, training=True)
        # fake latent
        E_hat = generator(z, training=False)
        H_fake = supervisor(E_hat, training=False)
        D_fake = discriminator(H_fake, training=True)
        # losses (logits)
        loss_d = bce_logits(tf.ones_like(D_real), D_real) + bce_logits(tf.zeros_like(D_fake), D_fake)
    grads = tape.gradient(loss_d, discriminator.trainable_variables)
    opt_d.apply_gradients(zip(grads, discriminator.trainable_variables))
    return loss_d

@tf.function
def generator_step(x):
    batch_size = tf.shape(x)[0]
    z = sample_z(batch_size)
    with tf.GradientTape(persistent=True) as tape:
        # generate sequence
        E_hat = generator(z, training=True)
        H_hat = supervisor(E_hat, training=True)
        # reconstruct to data space
        X_hat = recovery(H_hat, training=True)
        # discriminator output on fake
        D_fake = discriminator(H_hat, training=False)
        # losses
        g_loss_u = bce_logits(tf.ones_like(D_fake), D_fake)            # adversarial loss (unconditional)
        # supervised loss (encourage dynamics)
        H_real = embedder(x, training=False)
        g_loss_s = mse(H_real[:,1:,:], H_hat[:, :-1, :])
        # reconstruction loss from embedder & recovery (embedder may be trainable here)
        H_enc = embedder(x, training=True)
        X_tilde = recovery(H_enc, training=True)
        g_loss_recon = mse(x, X_tilde)
        # total
        total_g_loss = g_loss_u + LAMBDA_SUP * g_loss_s + LAMBDA_REC * g_loss_recon
    # apply gradients to generator, supervisor, optionally embedder & recovery
    gv = generator.trainable_variables + supervisor.trainable_variables + embedder.trainable_variables + recovery.trainable_variables
    grads = tape.gradient(total_g_loss, gv)
    opt_g.apply_gradients(zip(grads, gv))
    return total_g_loss, g_loss_u, g_loss_s, g_loss_recon

# ----------------- Training loop -----------------
print("=== Supervisor pretraining ===")
for epoch in range(1, EPOCHS_SUPERVISOR + 1):
    t0 = time.time()
    losses = []
    for batch in train_ds:
        l = supervised_step(batch)
        losses.append(l.numpy())
    if epoch % 5 == 0 or epoch==1:
        print(f"[Sup] Epoch {epoch}/{EPOCHS_SUPERVISOR}  loss={np.mean(losses):.6f}  time={time.time()-t0:.1f}s")

# Save a checkpoint after supervisor pretraining
manager.save(checkpoint_number=0)
print("Saved supervisor checkpoint.")

print("=== Adversarial training ===")
for epoch in range(1, EPOCHS_ADVERSARIAL + 1):
    t0 = time.time()
    d_losses, g_losses = [], []
    for batch in train_ds:
        d_l = discriminator_step(batch)
        g_l, g_adv, g_sup, g_rec = generator_step(batch)
        d_losses.append(d_l.numpy())
        g_losses.append(g_l.numpy())
    # logging
    print(f"[Adv] Epoch {epoch}/{EPOCHS_ADVERSARIAL}  D={np.mean(d_losses):.6f}  G={np.mean(g_losses):.6f}  time={time.time()-t0:.1f}s")
    # save checkpoint periodically
    if epoch % 10 == 0:
        manager.save()
        # quick MMD on a sample
        real_sample = train[:BATCH_SIZE]
        synth = None
        # use generator to synthesize BATCH_SIZE samples
        z = sample_z(BATCH_SIZE)
        Ehat = generator(z, training=False)
        Hhat = supervisor(Ehat, training=False)
        Xhat = recovery(Hhat, training=False).numpy()
        # inverse scale?
        # compute mmd on scaled arrays
        try:
            mm = mmd_rbf(real_sample, Xhat)
            print(f"  checkpoint saved. sample MMD: {mm:.6f}")
        except Exception:
            print("  checkpoint saved.")

# Save final weights in compatible filenames
embedder.save_weights(os.path.join(CKPT_DIR, "embedder_final.weights.h5"))
recovery.save_weights(os.path.join(CKPT_DIR, "recovery_final.weights.h5"))
generator.save_weights(os.path.join(CKPT_DIR, "generator_final.weights.h5"))
supervisor.save_weights(os.path.join(CKPT_DIR, "supervisor_final.weights.h5"))
discriminator.save_weights(os.path.join(CKPT_DIR, "discriminator_final.weights.h5"))
print("Saved final weights.")

# ----------------- Generation helper -----------------
def generate(n_samples):
    Z = tf.random.normal([n_samples, SEQ_LEN, Z_DIM])
    Ehat = generator(Z, training=False)
    Hhat = supervisor(Ehat, training=False)
    Xhat = recovery(Hhat, training=False).numpy()  # scaled values
    # inverse scale to original units
    with open(os.path.join(DATA_DIR, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    n, T, D = Xhat.shape
    X_inv = scaler.inverse_transform(Xhat.reshape(-1, D)).reshape(n, T, D)
    return X_inv

if __name__ == "__main__":
    # quick test generate before or after training if you want
    s = generate(5)
    print("Generated sample shapes:", s.shape)

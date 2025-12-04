# src/timegan_tf.py
import tensorflow as tf
from tensorflow.keras import layers, Model


# ---------- Helper: RNN Block (for reuse) ----------
def rnn_block(hidden_dim, num_layers=1, return_sequences=True, name_prefix="rnn", dropout=0.0):
    blocks = []
    for i in range(num_layers):
        gru = layers.GRU(
            hidden_dim,
            return_sequences=(return_sequences if i == num_layers - 1 else True),
            name=f"{name_prefix}_gru_{i}"
        )
        blocks.append(gru)
        if dropout > 0.0:
            blocks.append(layers.Dropout(dropout))
        blocks.append(layers.LayerNormalization())
    return blocks


# ---------- Embedder ----------
class Embedder(Model):
    """Embedder: X (batch, T, D) → H (batch, T, hidden)"""
    def __init__(self, input_dim, hidden_dim, num_layers=2, dropout=0.1):
        super().__init__()
        self.grus = rnn_block(hidden_dim, num_layers, return_sequences=True, name_prefix="embedder", dropout=dropout)

    def call(self, x, training=False):
        h = x
        for layer in self.grus:
            h = layer(h, training=training)
        return h  # (batch, T, hidden)


# ---------- Recovery ----------
class Recovery(Model):
    """Recovery: H (batch, T, hidden) → X_tilde (batch, T, D)"""
    def __init__(self, hidden_dim, output_dim, num_layers=2, dropout=0.1):
        super().__init__()
        self.grus = rnn_block(hidden_dim, num_layers, return_sequences=True, name_prefix="recovery", dropout=dropout)
        self.out = layers.TimeDistributed(layers.Dense(output_dim), name="recovery_out")

    def call(self, h, training=False):
        y = h
        for layer in self.grus:
            y = layer(y, training=training)
        return self.out(y)


# ---------- Generator ----------
class Generator(Model):
    """Generator: Z (batch, T, z_dim) → E_hat (batch, T, hidden)"""
    def __init__(self, z_dim, hidden_dim, num_layers=2, dropout=0.2):
        super().__init__()
        self.proj = tf.keras.Sequential([
            layers.TimeDistributed(layers.Dense(hidden_dim, activation='tanh')),
            layers.LayerNormalization()
        ])
        self.grus = rnn_block(hidden_dim, num_layers, return_sequences=True, name_prefix="generator", dropout=dropout)

    def call(self, z, training=False):
        h = self.proj(z)
        for layer in self.grus:
            h = layer(h, training=training)
        return h  # (batch, T, hidden)


# ---------- Supervisor ----------
class Supervisor(Model):
    """Supervisor: E_hat (batch, T, hidden) → H_hat (batch, T, hidden)"""
    def __init__(self, hidden_dim, num_layers=2, dropout=0.2):
        super().__init__()
        self.grus = rnn_block(hidden_dim, num_layers, return_sequences=True, name_prefix="supervisor", dropout=dropout)
        self.project = layers.TimeDistributed(layers.Dense(hidden_dim, activation='tanh'), name="supervisor_proj")

    def call(self, h, training=False):
        y = h
        for layer in self.grus:
            y = layer(y, training=training)
        return self.project(y)


# ---------- Discriminator ----------
class Discriminator(Model):
    """Discriminator: H (batch, T, hidden) → logit (batch, 1)"""
    def __init__(self, hidden_dim, num_layers=2, dropout=0.3):
        super().__init__()
        self.grus = [layers.GRU(hidden_dim, return_sequences=True, name=f"discriminator_gru_{i}")
                     for i in range(num_layers)]
        self.dropout = layers.Dropout(dropout)
        self.fc = layers.Dense(1, name="discriminator_fc")

    def call(self, h, training=False):
        y = h
        for g in self.grus:
            y = g(y, training=training)
            y = self.dropout(y, training=training)
        # Only use last timestep for classification
        y_last = y[:, -1, :]
        return self.fc(y_last)

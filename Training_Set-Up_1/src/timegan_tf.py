# src/timegan_tf.py
import tensorflow as tf
from tensorflow.keras import layers, Model

# Helper RNN block (GRU)
def rnn_block(hidden_dim, num_layers=1, return_sequences=True, name_prefix="rnn"):
    blocks = []
    for i in range(num_layers):
        blocks.append(layers.GRU(hidden_dim, return_sequences=(return_sequences if i==num_layers-1 else True),
                                 return_state=False, name=f"{name_prefix}_gru_{i}"))
    return blocks

class Embedder(Model):
    """Embedder: X (batch, T, D) -> H (batch, T, hidden)"""
    def __init__(self, input_dim, hidden_dim, num_layers=1):
        super().__init__()
        self.grus = [layers.GRU(hidden_dim, return_sequences=True, name=f"embedder_gru_{i}") for i in range(num_layers)]

    def call(self, x, training=False):
        h = x
        for g in self.grus:
            h = g(h, training=training)
        return h  # (batch, T, hidden)

class Recovery(Model):
    """Recovery: H (batch, T, hidden) -> X_tilde (batch, T, D)"""
    def __init__(self, hidden_dim, output_dim, num_layers=1):
        super().__init__()
        self.grus = [layers.GRU(hidden_dim, return_sequences=True, name=f"recovery_gru_{i}") for i in range(num_layers)]
        self.out = layers.TimeDistributed(layers.Dense(output_dim), name="recovery_out")

    def call(self, h, training=False):
        y = h
        for g in self.grus:
            y = g(y, training=training)
        return self.out(y)  # (batch, T, D)

class Generator(Model):
    """Generator: Z (batch, T, z_dim) -> E_hat (batch, T, hidden)"""
    def __init__(self, z_dim, hidden_dim, num_layers=1):
        super().__init__()
        self.proj = layers.TimeDistributed(layers.Dense(hidden_dim), name="gen_proj")
        self.grus = [layers.GRU(hidden_dim, return_sequences=True, name=f"generator_gru_{i}") for i in range(num_layers)]

    def call(self, z, training=False):
        h = self.proj(z)  # (batch, T, hidden)
        for g in self.grus:
            h = g(h, training=training)
        return h  # (batch, T, hidden)

class Supervisor(Model):
    """Supervisor: E_hat (batch, T, hidden) -> H_hat (batch, T, hidden). learns temporal transitions"""
    def __init__(self, hidden_dim, num_layers=1):
        super().__init__()
        self.grus = [layers.GRU(hidden_dim, return_sequences=True, name=f"supervisor_gru_{i}") for i in range(num_layers)]
        self.project = layers.TimeDistributed(layers.Dense(hidden_dim), name="supervisor_proj")

    def call(self, h, training=False):
        y = h
        for g in self.grus:
            y = g(y, training=training)
        return self.project(y)  # (batch, T, hidden)

class Discriminator(Model):
    """Discriminator: H (batch, T, hidden) -> logit (batch, 1)"""
    def __init__(self, hidden_dim, num_layers=1):
        super().__init__()
        self.gru = layers.GRU(hidden_dim, return_sequences=False, name="discriminator_gru")
        self.fc = layers.Dense(1, name="discriminator_fc")  # logits

    def call(self, h, training=False):
        y = self.gru(h, training=training)  # (batch, hidden)
        return self.fc(y)  # (batch, 1)

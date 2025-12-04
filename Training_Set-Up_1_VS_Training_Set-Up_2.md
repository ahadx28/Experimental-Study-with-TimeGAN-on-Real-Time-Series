# TimeGAN Training Setup Comparison

This document provides a detailed comparison between two training configurations for TimeGAN (Time-series Generative Adversarial Network) on household electricity consumption data.

## Overview

- **Training Set-Up 1**: Basic TimeGAN implementation with standard GAN training
- **Training Set-Up 2**: Enhanced TimeGAN with WGAN-GP (Wasserstein GAN with Gradient Penalty)

---

## Key Differences Summary

| Aspect | Set-Up 1 | Set-Up 2 |
|--------|----------|----------|
| **GAN Type** | Standard GAN | WGAN-GP (Wasserstein GAN with Gradient Penalty) |
| **Model Depth** | 1 GRU layer per component | 2 GRU layers per component |
| **Regularization** | None | Dropout + Layer Normalization |
| **Learning Rate** | Single LR (1e-4) | Component-specific LRs with decay |
| **Training Strategy** | Balanced G/D updates | 5 D updates per G update (n_critic=5) |
| **Training Duration** | 60 + 300 epochs | 80 + 400 epochs |
| **Monitoring** | Basic checkpointing | TensorBoard logging + validation monitoring |

---

## Detailed Comparison

### 1. Model Architecture

#### Set-Up 1: Basic Architecture
```python
# Simple RNN blocks
- Embedder: 1-layer GRU
- Recovery: 1-layer GRU
- Generator: 1-layer GRU
- Supervisor: 1-layer GRU
- Discriminator: 1-layer GRU + Dense
```

**Characteristics:**
- Minimal architecture complexity
- No regularization techniques
- Simple forward pass

#### Set-Up 2: Enhanced Architecture
```python
# Deeper networks with regularization
- Embedder: 2-layer GRU + Dropout (0.1) + LayerNorm
- Recovery: 2-layer GRU + Dropout (0.1) + LayerNorm
- Generator: 2-layer GRU + Dropout (0.2) + LayerNorm + tanh activation
- Supervisor: 2-layer GRU + Dropout (0.2) + LayerNorm + tanh activation
- Discriminator: 2-layer GRU + Dropout (0.3)
```

**Characteristics:**
- Increased model capacity with deeper networks
- Dropout for better generalization (varying rates: 0.1-0.3)
- Layer normalization for training stability
- Improved activation functions (tanh) for better gradient flow

**Code File:** `src/timegan_tf.py`
- Set-Up 1: 73 lines
- Set-Up 2: 101 lines (+38% expansion)

---

### 2. Training Configuration

#### Set-Up 1: Standard GAN Training
```python
# Hyperparameters
LR = 1e-4                    # Single learning rate
Z_DIM = 32                   # Noise dimension
EPOCHS_SUPERVISOR = 60       # Pretraining epochs
EPOCHS_ADVERSARIAL = 300     # Adversarial epochs
LAMBDA_SUP = 100.0          # Supervised loss weight
LAMBDA_REC = 10.0           # Reconstruction loss weight
LAMBDA_MOMENT = 100.0       # Moment matching loss weight
```

**Training Strategy:**
- Binary cross-entropy loss for discriminator
- Balanced generator/discriminator updates (1:1 ratio)
- Fixed learning rate throughout training
- Basic loss weighting scheme

#### Set-Up 2: WGAN-GP Training
```python
# Hyperparameters
LR_D = 1e-4                  # Discriminator learning rate
LR_G = 2e-4                  # Generator learning rate
LR_E = 5e-4                  # Embedder learning rate
Z_DIM = 16                   # Noise dimension (reduced)
EPOCHS_SUPERVISOR = 80       # Pretraining epochs
EPOCHS_ADVERSARIAL = 400     # Adversarial epochs
N_CRITIC = 5                 # D updates per G update
LAMBDA_SUP = 3.0            # Supervised loss weight (reduced)
LAMBDA_REC = 10.0           # Reconstruction loss weight
LAMBDA_GP = 10.0            # Gradient penalty weight
```

**Training Strategy:**
- Wasserstein distance instead of binary cross-entropy
- Gradient penalty for Lipschitz constraint enforcement
- Multiple discriminator updates per generator update (5:1 ratio)
- Component-specific learning rates
- Exponential learning rate decay (decay_rate=0.95, decay_steps=1000)
- Extended training duration for better convergence

**Code File:** `src/train_timegan_adversarial_tf.py`
- Set-Up 1: 241 lines
- Set-Up 2: 567 lines (+135% expansion)

---

### 3. Data Preprocessing

#### Set-Up 1: Basic Preprocessing
```python
def preprocess_household(input_path, out_dir, seq_len=168, stride=24):
    # Hourly resampling
    # Interpolation
    # Basic normalization
```

**Features:**
- Stride of 24 hours (fewer overlapping sequences)
- Simple interpolation for missing values
- Standard scaling

#### Set-Up 2: Enhanced Preprocessing
```python
def preprocess_household(input_path, out_dir, seq_len=168, stride=12, smooth=True):
    # Hourly resampling
    # Interpolation
    # Rolling mean smoothing (window=3)
    # Normalization
```

**Features:**
- Stride of 12 hours (more training data with better overlap)
- Optional smoothing to reduce noise/spikes
- Rolling mean filter (3-hour window, centered)
- More robust data handling

**Code File:** `src/preprocess_electricity.py`
- Set-Up 1: 96 lines
- Set-Up 2: 130 lines (+35% expansion)

---

### 4. Monitoring and Evaluation

#### Set-Up 1: Basic Monitoring
- Simple checkpoint saving
- Loss printing to console
- Manual evaluation required

#### Set-Up 2: Advanced Monitoring
```python
# TensorBoard Integration
train_log_dir = f'logs/wgan_gp/{current_time}'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)

# Validation Monitoring
VALIDATION_EVAL_EVERY = 5    # Evaluate every 5 epochs
VAL_MMD_PATIENCE = 12        # Early stopping patience
```

**Features:**
- Real-time TensorBoard logging
- Comprehensive metrics tracking:
  - Discriminator loss (real, fake, gradient penalty)
  - Generator loss
  - Reconstruction loss
  - Supervised loss
  - Learning rates
- Validation set monitoring with MMD (Maximum Mean Discrepancy)
- Early stopping based on validation performance
- Timestamped logging directories

---

### 5. Loss Functions and Objectives

#### Set-Up 1: Standard GAN Losses
```python
# Discriminator Loss
D_loss_real = BCE(D(real), ones)
D_loss_fake = BCE(D(fake), zeros)
D_loss = D_loss_real + D_loss_fake

# Generator Loss
G_loss_adv = BCE(D(fake), ones)
G_loss_total = G_loss_adv + λ_sup * L_sup + λ_rec * L_rec + λ_moment * L_moment
```

#### Set-Up 2: WGAN-GP Losses
```python
# Discriminator Loss (Wasserstein)
D_loss_real = -mean(D(real))
D_loss_fake = mean(D(fake))
D_loss_gp = λ_gp * gradient_penalty(D, real, fake)
D_loss = D_loss_real + D_loss_fake + D_loss_gp

# Generator Loss
G_loss_adv = -mean(D(fake))
G_loss_total = G_loss_adv + λ_sup * L_sup + λ_rec * L_rec
```

**WGAN-GP Advantages:**
- More stable training (no mode collapse)
- Meaningful loss values (Wasserstein distance)
- Gradient penalty enforces Lipschitz constraint
- Better convergence properties

---

### 6. Code Organization and Features

#### Set-Up 1
**Python Scripts:**
- `timegan_tf.py` (73 lines)
- `train_recon_tf.py` (102 lines)
- `train_timegan_adversarial_tf.py` (241 lines)
- `preprocess_electricity.py` (96 lines)
- `generate_and_save.py` (66 lines)
- `evaluate_synth.py` (66 lines)
- `evaluate_synth_fixed.py` (147 lines)
- `save_recon_artifacts.py` (60 lines)

**Total:** 851 lines

#### Set-Up 2
**Python Scripts:**
- `timegan_tf.py` (101 lines) - Enhanced models
- `train_recon_tf.py` (115 lines) - Improved training
- `train_timegan_adversarial_tf.py` (567 lines) - WGAN-GP implementation
- `preprocess_electricity.py` (130 lines) - Enhanced preprocessing
- `generate_and_save.py` (91 lines) - Extended generation
- `evaluate_synth.py` (66 lines)
- `evaluate_synth_fixed.py` (155 lines)
- `save_recon_artifacts.py` (79 lines)
- `plot_recon_artifacts.py` (5 lines) - New plotting utility

**Total:** 1,309 lines (+53% more code)

---

### 7. Additional Improvements in Set-Up 2

#### Checkpoint Management
- More robust checkpoint saving with optimizer state preservation
- JSON-based metadata storage for training configuration
- Better recovery from interrupted training sessions

#### Gradient Penalty Implementation
```python
def gradient_penalty(discriminator, real_data, fake_data):
    alpha = tf.random.uniform([batch_size, 1, 1])
    interpolated = alpha * real_data + (1 - alpha) * fake_data
    
    with tf.GradientTape() as tape:
        tape.watch(interpolated)
        pred = discriminator(interpolated, training=True)
    
    gradients = tape.gradient(pred, interpolated)
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2]))
    gp = tf.reduce_mean(tf.square(slopes - 1.0))
    return gp
```

#### Learning Rate Schedules
```python
lr_schedule_d = tf.keras.optimizers.schedules.ExponentialDecay(
    LR_D, decay_steps=1000, decay_rate=0.95, staircase=True)
lr_schedule_g = tf.keras.optimizers.schedules.ExponentialDecay(
    LR_G, decay_steps=1000, decay_rate=0.95, staircase=True)
lr_schedule_e = tf.keras.optimizers.schedules.ExponentialDecay(
    LR_E, decay_steps=1000, decay_rate=0.95, staircase=True)
```

---

## Performance Implications

### Set-Up 1: Strengths
- ✅ Simpler implementation, easier to understand
- ✅ Faster training per epoch
- ✅ Lower memory footprint
- ✅ Good for initial experiments and quick iterations

### Set-Up 1: Limitations
- ⚠️ Potential training instability (mode collapse)
- ⚠️ Less sophisticated model capacity
- ⚠️ Limited monitoring and diagnostics
- ⚠️ May underfit complex temporal patterns

### Set-Up 2: Strengths
- ✅ Superior training stability with WGAN-GP
- ✅ Better model capacity with deeper networks
- ✅ Comprehensive monitoring and early stopping
- ✅ More robust to hyperparameter choices
- ✅ Better generalization with regularization
- ✅ Extended training for better convergence
- ✅ Component-specific learning rates for balanced training

### Set-Up 2: Considerations
- ⚠️ Higher computational cost (deeper networks, more epochs)
- ⚠️ Longer training time (5× discriminator updates)
- ⚠️ More hyperparameters to tune
- ⚠️ Higher memory requirements

---

## Recommended Use Cases

### Choose Set-Up 1 if:
- You need quick experiments and rapid prototyping
- Computational resources are limited
- Working with simpler time series patterns
- Learning TimeGAN fundamentals
- Need baseline results for comparison

### Choose Set-Up 2 if:
- Production-quality synthetic data is required
- Training stability is crucial
- Working with complex, multi-variate time series
- Computational resources are available
- Need state-of-the-art generation quality
- Require comprehensive monitoring and diagnostics

---

## Migration Path

To migrate from Set-Up 1 to Set-Up 2:

1. **Update model definitions** (`timegan_tf.py`)
   - Add dropout and layer normalization
   - Increase number of GRU layers to 2

2. **Modify training script** (`train_timegan_adversarial_tf.py`)
   - Implement WGAN-GP loss functions
   - Add gradient penalty computation
   - Set up component-specific optimizers with LR schedules
   - Add TensorBoard logging
   - Implement validation monitoring

3. **Update preprocessing** (`preprocess_electricity.py`)
   - Reduce stride to 12 (from 24)
   - Add smoothing option
   - Enhanced data cleaning

4. **Adjust hyperparameters**
   - Set `N_CRITIC = 5`
   - Use different learning rates: `LR_D=1e-4, LR_G=2e-4, LR_E=5e-4`
   - Reduce `LAMBDA_SUP` to 3.0
   - Increase training epochs to 400

---

## File Structure

```
Training_Set-Ups/
├── requirements.txt                    # Shared dependencies
├── Training_Set-Up_1/
│   ├── data/
│   │   └── raw/
│   │       └── link_to_household_power_consumption_data.txt
│   └── src/
│       ├── timegan_tf.py               # Basic model definitions
│       ├── train_recon_tf.py           # Reconstruction pretraining
│       ├── train_timegan_adversarial_tf.py  # Standard GAN training
│       ├── preprocess_electricity.py   # Basic preprocessing
│       ├── generate_and_save.py        # Generation script
│       ├── evaluate_synth.py           # Evaluation utilities
│       ├── evaluate_synth_fixed.py     # Extended evaluation
│       └── save_recon_artifacts.py     # Artifact saving
│       └── [various .ipynb notebooks]  # Analysis notebooks
│
└── Training_Set-Up_2/
    ├── data/
    │   └── raw/
    │       └── link_to_household_power_consumption_data.txt
    └── src/
        ├── timegan_tf.py               # Enhanced model definitions
        ├── train_recon_tf.py           # Improved reconstruction training
        ├── train_timegan_adversarial_tf.py  # WGAN-GP training
        ├── preprocess_electricity.py   # Enhanced preprocessing
        ├── generate_and_save.py        # Extended generation
        ├── evaluate_synth.py           # Evaluation utilities
        ├── evaluate_synth_fixed.py     # Extended evaluation
        ├── save_recon_artifacts.py     # Enhanced artifact saving
        ├── plot_recon_artifacts.py     # NEW: Plotting utility
        └── [various .ipynb notebooks]  # Analysis notebooks
```

---

## Dependencies

Both setups share the same dependencies (see `requirements.txt`):
```
tensorflow>=2.x
numpy
pandas
scikit-learn
matplotlib
seaborn
```

Additional for Set-Up 2 monitoring:
- TensorBoard (included with TensorFlow)

---

## References

**TimeGAN Original Paper:**
- Yoon, J., Jarrett, D., & Van der Schaar, M. (2019). Time-series generative adversarial networks. *Advances in neural information processing systems*, 32.

**WGAN-GP:**
- Gulrajani, I., Ahmed, F., Arjovsky, M., Dumoulin, V., & Courville, A. C. (2017). Improved training of wasserstein gans. *Advances in neural information processing systems*, 30.

**Dataset:**
- UCI Machine Learning Repository: Individual household electric power consumption Data Set

---

## Conclusion

**Set-Up 1** provides a solid foundation for understanding TimeGAN and conducting initial experiments with reasonable results. It's ideal for learning, quick prototyping, and establishing baselines.

**Set-Up 2** represents a production-ready implementation with state-of-the-art training techniques (WGAN-GP), enhanced model architecture, and comprehensive monitoring. It's designed for generating high-quality synthetic time series data with superior stability and convergence properties.

The choice between setups depends on your specific requirements, computational resources, and whether you prioritize rapid experimentation or production-quality results.

---

## Contact & Contributions

For questions, issues, or contributions related to these training setups, please refer to the repository's issue tracker or contact the maintainers.

**Last Updated:** December 2025

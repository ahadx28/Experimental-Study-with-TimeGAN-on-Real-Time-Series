import numpy as np, matplotlib.pyplot as plt
orig = np.load("outputs/checkpoints/recon/recon_examples_orig.npy")
rec  = np.load("outputs/checkpoints/recon/recon_examples_rec.npy")
plt.figure(figsize=(10,3))
plt.plot(orig[0,:,0], label='orig'); plt.plot(rec[0,:,0], label='rec'); plt.legend(); plt.show()

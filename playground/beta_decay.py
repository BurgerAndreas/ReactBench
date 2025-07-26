import numpy as np
import matplotlib.pyplot as plt

# Create k values from a small positive number to 10^6
k = np.logspace(0, 3, 10000)  # 10000 points from 0.001 to 1,000,000

# Calculate beta = 1.0 - k^(-0.8)
beta = 1.0 - k ** (-0.8)

# Create the plot
plt.figure(figsize=(10, 6))
plt.semilogx(k, beta)
plt.xlabel("k")
plt.ylabel("β = 1.0 - k^(-0.8)")
plt.title("Beta Decay Function: β = 1.0 - k^(-0.8)")
plt.grid(True, alpha=0.3)
# plt.ylim(-0.1, 1.1)  # Set y-axis limits for better visualization

# Show the plot
plt.tight_layout()
fname = "playground/plots/beta_decay.png"
plt.savefig(fname)
print(f"Saved plot to {fname}")
plt.close()

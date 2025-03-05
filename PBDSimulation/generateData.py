import numpy as np
from Physics import muscleForce
import matplotlib.pyplot as plt

def generate_muscle_force_data(lm_range, activation_range, pennation_range, batch_size=100000):
    L, A, P = np.meshgrid(lm_range, activation_range, pennation_range, indexing="ij")

    # Flatten arrays for batch processing
    L_flat, A_flat, P_flat = L.ravel(), A.ravel(), P.ravel()
    F_flat = np.zeros_like(L_flat, dtype=np.float32)

    # Process in batches
    for i in range(0, L_flat.shape[0], batch_size):
        batch_indices = slice(i, i + batch_size)
        F_flat[batch_indices] = np.array([muscleForce(L_flat[j], A_flat[j], P_flat[j]) for j in range(i, min(i+batch_size, L_flat.shape[0]))])

    # Reshape back to original shape
    F = F_flat.reshape(L.shape)
    return L, A, P, F


sample_size = 2000  # Reduced for faster computation

# Define normal distribution parameters
mu = 2.0  # Mean around the center of 0.5 to 3.0
sigma = 0.8  # Standard deviation to control spread
lMtilde = np.random.normal(mu, sigma, sample_size)
lMtilde = np.clip(lMtilde, 0.0, 5.0)
a = np.linspace(0.0, 1.0, 1000)
p = np.linspace(0.0, 0.7, 1000)

L, A, P, F = generate_muscle_force_data(lMtilde, a, p)

plt.hist(lMtilde, bins=20, edgecolor='k')
plt.title('Histogram of lMtilde')
plt.xlabel('lMtilde values')
plt.ylabel('Frequency')
plt.show()

# Save L, A, F as .npz file
np.savez('TrainingData/lM_act_pen_force.npz', L=L, A=A, P=P, F=F)
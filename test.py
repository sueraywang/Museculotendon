import numpy as np
sample_size = 5
lMtilde = np.linspace(1, 5, sample_size)
lTtilde = np.linspace(1, 5, sample_size)
X, Y = np.meshgrid(lMtilde, lTtilde)

X_train = np.vstack([X.ravel(), Y.ravel()]).T

print(X_train)
for x, y in zip(X.ravel(), Y.ravel()):
    print(x, y)
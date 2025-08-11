import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from mpl_toolkits.mplot3d import Axes3D

# Reproducibility
np.random.seed(42)

# Generate synthetic data
n_samples = 100
x1 = np.random.uniform(-3, 3, n_samples)
x2 = np.random.uniform(-3, 3, n_samples)
x = np.column_stack((x1, x2))
y = np.sin(x1) * np.cos(x2) + 0.1 * np.random.rand(n_samples)

# Degrees to test
degrees = [1, 2, 3]

# Create grid for surface plotting
x1_min, x1_max = x1.min() - 1, x1.max() + 1
x2_min, x2_max = x2.min() - 1, x2.max() + 1
xx1, xx2 = np.meshgrid(
    np.linspace(x1_min, x1_max, 50),
    np.linspace(x2_min, x2_max, 50)
)
x_grid = np.c_[xx1.ravel(), xx2.ravel()]

# Create figure
fig = plt.figure(figsize=(15, 5))

# Fit and plot for each degree
for i, degree in enumerate(degrees, 1):
    polyreg = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    polyreg.fit(x, y)
    y_pred = polyreg.predict(x_grid).reshape(xx1.shape)
    
    ax = fig.add_subplot(1, len(degrees), i, projection='3d')
    ax.scatter(x1, x2, y, color='blue', label='Data points')
    ax.plot_surface(xx1, xx2, y_pred, cmap='cool', alpha=0.7)
    ax.set_title(f'Polynomial Degree {degree}')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('y')

plt.tight_layout()
plt.show()

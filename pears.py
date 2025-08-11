from scipy.stats import pearsonr
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(42)
theta = np.linspace(0, 6 * np.pi, 500)
r=np.linspace(0,50, 500)+np.random.normal(0, 2, 500)
x = r * np.cos(theta)
y = r * np.sin(theta)

r_value,p_value = pearsonr(x, y)
plt.figure(figsize=(7,7))
plt.scatter(x, y,c=theta, cmap='brg',s=5,alpha=0.5)
plt.title(f"Pearson correlation: {r_value:.3f}",fontsize=16)
plt.colorbar(label='angle Theta')
plt.show()
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
np.random.seed(42)
x=np.random.rand(100)*10
# n=3x+5 
y=3 * x + 5 + np.random.rand(100)*2

m, c=np.polyfit(x,y,1)
print(f"Slope(m): {m:.2f}")
print(f"intercept (c): {c:.2f}")

new_x=9
print(f"predicted y for x={new_x}: {m*new_x+c:.2f}")
plt.scatter(x,y,color='blue',alpha=0.6,label='datapoints')
plt.plot(m*x+c,color='red',lw=2,label='best fitted line')
plt.xlabel('x-values')
plt.ylabel('y-label')
plt.legend()
plt.show()


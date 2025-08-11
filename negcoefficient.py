import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

np.random.seed(42)
x=np.random.rand(100,1)*10
y=-2 * x + np.random.normal(0,1,(100,1))

x=x.reshape(-1,1)
y=y.ravel()

model=LinearRegression()
model.fit(x,y)

correlation=np.corrcoef(x.ravel(),y)[0,1]

plt.scatter(x,y,color='blue',label='Data points')
plt.plot(x,model.predict(x),color='red')
plt.xlabel("x")
plt.ylabel("y")
plt.title("negitive correlation")
plt.legend()
plt.show()

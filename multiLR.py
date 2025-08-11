import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

x=np.array([[5,2],[3,1],[8,3],[2,0],[6,2]])
y=np.array([72,50,90,30,85])

#train the model
model=LinearRegression()
model.fit(x,y)

y_pred=model.predict(x)

plt.scatter(y,y_pred,color='blue')
plt.plot([y.min(),y.max()],[y.min(),y.max()])
plt.show()
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
np.random.seed(42)
x,y=make_regression(n_samples=100,n_features=10,noise=15)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
ridge=Ridge(1.0)
ridge.fit(x_train,y_train)
y_pred=ridge.predict(x_test)
print("coefficient:",ridge.coef_)
print("MSE",mean_squared_error(y_test,y_pred))
x1_test=x_test[:,0]
x2_test=x_test[:,1]
fig=plt.figure(figsize=(10,8))
ax=fig.add_subplot(111,projection='3d')
ax.scatter(x1_test,x2_test,y_test,color='blue',label='Actual',alpha=0.8)
ax.scatter(x1_test,x2_test,y_pred,color='red',label='predicted',alpha=0.8)
x1_range=np.linspace(x1_test.min(),x1_test.max(),20)
x2_range=np.linspace(x2_test.min(),x2_test.max(),20)
x1_grid,x2_grid=np.meshgrid(x1_range,x2_range)
x_grid=np.zeros((x1_grid.size,x.shape[1]))
x_grid[:,0]=x1_grid.ravel()
x_grid[:,1]=x2_grid.ravel()
for i in range(2,x.shape[1]):
    x_grid[:,i]=np.mean(x_test[:,i])
y_grid=ridge.predict(x_grid)
y_grid=y_grid.reshape(x1_grid.shape)
ax.plot_surface(x1_grid,x2_grid,y_grid,color='green',alpha=0.8,rstride=1,cstride=1)
ax.set_xlabel('feature 1')
ax.set_ylabel('feature 2')
ax.set_title("Ridge-3D")
ax.legend()
def rotate(angle):
    ax.view_init(azim=angle)
ani=FuncAnimation(fig,rotate,frames=np.arange(0,360,2),interval=50)
plt.show()
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from mpl_toolkits.mplot3d import Axes3D
np.random.seed(42)
x= np.sort(5 *np.random.rand(80,1), axis=0)
y= np.sin(x).ravel()+np.random.normal(0,0.1,x.shape[0])
x=x.reshape(-1,1)
degrees=[1,2,3,4]
plt.figure(figsize=(10,6))
for degree in degrees:
    preg=make_pipeline(PolynomialFeatures(degree),LinearRegression())
    preg.fit(x,y)
    x_smooth=np.linspace(x.min(),x.max(), 300).reshape(-1,1)
    y_smooth=preg.predict(x_smooth)
    plt.plot(x_smooth, y_smooth)
plt.legend()
plt.show()
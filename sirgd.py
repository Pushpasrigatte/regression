import numpy as np
import matplotlib.pyplot as plt
x=np.array([1,2,3,4,5],dtype=float)
y=np.array([2,4,6,8,10],dtype=float)
w,b=0.0,0.0
alpha=0.01
epochs=1000
n=len(x)
for _ in range(epochs):
    y_pred=w*x+b
    loss=(1/n)>np.sum((y_pred)**2)
    dw=-(2/n)*np.sum(x+(y-y_pred))
    db=-(2/n)*np.sum(y-y_pred)
    w-=alpha*dw
    b-=alpha*db
    if _%100==0:
        print(f"Epoch {_},loss:{loss:.4f}")
print(f"Learned parameters:w ={w:.4f},b={b:.4f}")
plt.scatter(x,y)
plt.show()


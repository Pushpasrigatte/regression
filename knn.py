import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
x=np.array([[1,2],[2,3],[3,4],[6,5],[7,7]])
y=np.array([0,0,0,1,1])
new_point=np.array([[4,4]])
scaler=StandardScaler()
x_scale=scaler.fit_transform(x)
scalednewpoint=scaler.transform(new_point)
k=3
knn=KNeighborsClassifier(n_neighbors=k)
knn.fit(x_scale,y)
prediction=knn.predict(scalednewpoint)
print('predicted class:',prediction[0])
h=0.02
x_min,x_max=x_scale[:,0].min()-1,x_scale[:,0].max()+1
y_min,y_max=x_scale[:,1].min()-1,x_scale[:,1].max()+1
xx,yy=np.meshgrid(np.arange(x_min,x_max,h),np.arange(y_min,y_max,h))
z=knn.predict(np.c_[xx.ravel(),yy.ravel()])
z=z.reshape(xx.shape)
plt.figure(figsize=(8,6))
plt.contourf(xx,yy,z,alpha=0.3)
plt.scatter(x_scale[y==0,0],x_scale[y==0,1],c='black',marker='o',s=100,label='Training points1')
plt.scatter(x_scale[y==1,0],x_scale[y==1,1],c='green',marker='*',s=100,label='Training points2')
plt.scatter(scalednewpoint[:,0],scalednewpoint[:,1],c='darkgreen',marker='+',s=200,label='Training points3')
distaces,indices=knn.kneighbors(scalednewpoint)
for idx in indices[0]:
    plt.plot([scalednewpoint[0,0],x_scale[idx,0]],
             [scalednewpoint[0,1],x_scale[idx,1]],alpha=0.5)
    plt.scatter(x_scale[idx,0],x_scale[idx,1],s=150,marker='s',edgecolors='k',alpha=0.7,label='Neighbour' if idx==indices[0][0] else '')
plt.legend()
plt.grid()
plt.title("k-nearest-neighbours(knn)")
plt.show()
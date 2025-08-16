import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score,classification_report
import matplotlib.colors as mcolors

sns.set_style('whitegrid')
data={'hours':[2,4,6,8,10,3,5,7,9,1],
      'score':[90,77,45,65,43,78,90,98,65,32],
       'pass':[0,0,0,0,1,1,1,1,1,0]} 
df=pd.DataFrame(data)
x=df[['hours','score']]
y=df[['pass']]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,
                                               random_state=42)
model=LogisticRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
accuracy=accuracy_score(y_test,y_pred)
print(f"accuracy:{accuracy:.2f}")
print('\n classification report:')
print(classification_report(y_test,y_pred))
new_student=np.array([[5,70]])
prediction=model.predict(new_student)
print("\n prediction of new student(hrs=5,score=70):{'pass' if prediction[0]==1 else 'Fail}")
plt.figure(figsize=(10,6))
scatter=sns.scatterplot(x='hours',y='score',hue='pass',style='pass',size='pass',data=df,
                        palette='hot',sizes=(50,150))
x_min,x_max=x['hours'].min()-1,x['hours'].max()+1
y_min,y_max=x['score'].min()-1,x['score'].max()+5
xx,yy=np.meshgrid(np.arange(x_min,x_max,0.1),np.arange(y_min,y_max,0.1))
z=model.predict(np.c_[xx.ravel(),yy.ravel()])
z=z.reshape(xx.shape)

plt.contour(xx,yy,z,alpha=0.3,cmap='plasma')
plt.legend()
plt.show()

cm=confusion_matrix(y_test,y_pred)
uc=['red','blue','green','orange']
custom_cmap=mcolors.ListedColormap(uc)
sns.heatmap(cm,cmap=custom_cmap,cbar=False)
plt.show()

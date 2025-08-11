''' from sklearn.linear_model import LinearRegression
models-linear/ridge/lasso/logistic...
sklearn.linear_model
LinearRegression-
parameter calling ->fit_intercept=True
normalize
n_job-threads
positive=False

training model:
model.fit()
shape(samples,features)
coefficient-whether it is capable or not(ml car are unable to jump)

model.predict()
plynomial features-
1-doubles
reshape

'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
hours=np.array([1,2,3,4,5]).reshape(-1,1)
marks=np.array([2,4,6,4,5])
model=LinearRegression()
model.fit(hours,marks)

new_hours=np.array([[6]])
prediction=model.predict(new_hours)
print(f"predicted marks for 6 hours:{prediction[0]:.2f}")

plt.scatter (hours,marks,color='blue',label="original data")
plt.plot(hours,model.predict(hours),color='red',label='Regression line')
plt.xlabel("hours of study")
plt.ylabel("scored marks")
plt.legend()
plt.show()

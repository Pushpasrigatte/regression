import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
sns.set_style('whitegrid')
data={'hours':[2,4,6,8,10,3,5,7,9,1],
      'score':[90,77,45,65,43,78,90,98,65,32],
       'pass':[0,0,0,0,1,1,1,1,1,0]} 
df = pd.DataFrame(data) 
plt.figure(figsize=(8,4))
sns.scatterplot(x='hours', y='score', hue='pass',data=df,style='pass',palette='coolwarm')
plt.suptitle("pairplot for pass/fail", y=0.2)
plt.show()

# Boxplot
plt.figure(figsize=(6,4))
sns.boxplot(x='pass', y='score', data=df)
plt.suptitle('Boxplot of Scores by Pass/Fail')
plt.show()

# Histogram
plt.figure(figsize=(6,4))
sns.histplot(data=df,x='hours', hue='pass',multiple='stack',palette='viridis')
plt.suptitle('Histogram of Scores')
plt.show()

# Bar plot
plt.figure(figsize=(6,4))
sns.barplot(x='pass', y='score', data=df, estimator='mean')
plt.suptitle('Average Score by Pass/Fail')
plt.show()

correlation_matrix=df.corr()
sns.heatmap(correlation_matrix,annot=True,cmap='cool',vmin=1,vmax=1)
plt.show()
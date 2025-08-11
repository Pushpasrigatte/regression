import matplotlib.pyplot as plt
import pickle

fig,ax=plt.subplots()
ax.plot([1,2,3,4],[11,22,33,44],label="Line-1")
ax.set_title("sample plot")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.legend()

with open('plot.pkl','wb') as f:
    pickle.dump(fig,f)

with open('plot.pkl','rb') as f:
    loaded_fig=pickle.load(f)

plt.show(block=True)
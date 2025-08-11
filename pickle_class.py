import pickle 
import numpy as np
class student:
    def __init__(self,name,age):
        self.name=name
        self.age=age
    def hi(self):
        return f"Hello\nmy name is {self.name}."
    
s=student('bujji',34)
with open('student.pkl','wb') as f:
    pickle.dump(s,f)
with open ('student.pkl','rb') as f:
    loaded_data=pickle.load(f)

print(loaded_data.name)
print(loaded_data.hi())        
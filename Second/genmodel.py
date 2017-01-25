import numpy as np
shape = (2,10,10,2)
weights = []
biases = []
from time import time
np.random.seed(int(time()))
for x in range(len(shape)-1):
    weights.append((np.random.randn(shape[x],shape[x+1]))/np.sqrt(shape[x]))
    biases.append(np.zeros((1,shape[x+1])))
model={"weights":weights,"biases":biases}
import pickle
pickle.dump(model,open("model.txt","w"))

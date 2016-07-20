import sklearn
import numpy as np
from time import time
size = 200
np.random.seed(int(time()))
import sklearn.datasets as datasets
x,y = datasets.make_moons(size,noise=0.20)
actY = np.zeros([size,2])
actY[range(size),y]=1
dataset = {"x":x,"y":actY}
import pickle
pickle.dump(dataset,open("data.txt","w"))

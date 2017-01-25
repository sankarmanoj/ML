dimension = (5,3)   #dimension(layers,nodes)
import numpy
import math
import pickle
a = numpy.random.rand(dimension[0]-1,dimension[1],dimension[1])
a = a*10
b = numpy.random.rand(dimension[0]-1,dimension[1],dimension[1])
b = b*-10
weights = a+b
print weights
pickle.dump(weights,open("weights.txt","w"))

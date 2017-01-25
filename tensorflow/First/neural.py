dimension = (100,3)   #dimension(layers,nodes)
import numpy
import math
import pickle
weights = numpy.array(pickle.load(open("weights.txt","r")))
print "Iterations"
inputarr = numpy.array([1,2,3],)
output = numpy.array([0,1,-1])
tanh = numpy.vectorize(math.tanh)
for weight in weights:
    inputarr = numpy.dot(weight,inputarr)
    inputarr =  tanh(inputarr)
print inputarr
err = output-inputarr
err = numpy.square(err)/2
print err
print err.sum()

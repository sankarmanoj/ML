import numpy as np
import pickle
shape = (16384,10,10,2)
model = pickle.load(open("model.txt","r"))
cars = pickle.load(open("cars.dat","r"))
faces = pickle.load(open("faces.dat","r"))
print cars.shape
print faces.shape
facecost = np.zeros([len(faces),2])
facecost[range(len(faces)),1]=1
carcost = np.zeros([len(cars),2])
carcost[range(len(cars)),0]=1
from copy import copy
reg_lambda = 0.01
epsilon = 0.0001
biases = model["biases"]
weights = model["weights"]
def activation(val):
    return np.tanh(val)
def dactiv(val):
    a  = np.power(np.tanh(val),2)
    return 1-a
# tries = input("Enter number of tries")
def calculate(inputValue):
    # inter=[]
    temp = inputValue
    # inter.append(temp)
    for weight in weights:
        temp = np.dot(temp,weight)
        temp = activation(temp)
        # inter.append(temp)
    return temp
def train(inputValue,actualValue):
    activ=[]
    temp = inputValue
    activ.append(temp)
    for weight in weights:
        temp = np.dot(temp,weight)
        temp = activation(temp)
        activ.append(temp)
    deltaOut = temp - actualValue
    deltatemp = np.copy(deltaOut)
    backActiv = copy(activ)
    backActiv.reverse()
    backWeight = copy(weights)
    backWeight.reverse()
    deltaWeights = []
    for x in range(len(backWeight)):
        deltatemp = np.multiply(deltatemp,dactiv(backActiv[x]))
        deltaWeights.append(deltatemp)
        deltatemp=np.dot(deltatemp,backWeight[x].T)
    deltaWeights.reverse()
    reducdDw = []
    for x in range(len(deltaWeights)):
        reducdDw.append(np.average(deltaWeights[x],axis=0))
    # print "Delta Weight"
    # for x in reducdDw:
    #     print x
    # print "Weigths"
    # for x in  weights:
    #     print x
    for x in range(len(weights)):
        biases[x]-=epsilon*reducdDw[x]
        wTemp = np.dot(activ[x].T,deltaWeights[x])
        weights[x]-=epsilon*wTemp



def cost(inputValue,actualValue):
    output = calculate(inputValue)
    temp = output - actualValue
    temp = np.power(temp,2)
    cost = np.sum(temp)
    return cost
def costMatrix(inputValue,actualValue):
    output = calculate(inputValue)
    temp = output - actualValue
    temp = np.power(temp,2)
    temp = np.sum(temp,axis=1)
    return temp
def save():
    nModel = {}
    nModel["biases"]=biases
    nModel["weights"]=weights
    pickle.dump(nModel,open("model.txt","w"))
def load():
    model = pickle.load(open("model.txt","r"))

#
# model["weights"] = weights
# model["biases"] = biases
# pickle.dump(model,open("model.txt","w"))
if __name__ =="__main__":
    times = input("Times")
    oldcost=10000
    from math import sqrt
    sqtime = int(sqrt(times))
    for x in range(sqtime):
        for x in range(sqtime):
            train(faces,facecost)
            train(cars,carcost)
        save()
        Tcost = cost(faces,facecost)+cost(cars,carcost)
        print "Cost = ",Tcost

        # load()

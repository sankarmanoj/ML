import numpy as np
import pickle
shape = (2,10,10,2)
model = pickle.load(open("model.txt","r"))
data = pickle.load(open("data.txt","r"))
inX = data["x"]
inY = data["y"]
from copy import copy
reg_lambda = 0.01
epsilon = 0.001
biases = model["biases"]
weights = model["weights"]
def sigmoid(val):
    return np.tanh(val)
def dsig(val):
    a  = np.power(np.tanh(val),2)
    return 1-a
# tries = input("Enter number of tries")
def calculate(inputValue):
    # inter=[]
    temp = inputValue
    # inter.append(temp)
    for weight in weights:
        temp = np.dot(temp,weight)
        temp = sigmoid(temp)
        # inter.append(temp)
    return temp
def train(inputValue,actualValue):
    activ=[]
    temp = inputValue
    activ.append(temp)
    for weight in weights:
        temp = np.dot(temp,weight)
        temp = sigmoid(temp)
        activ.append(temp)
    print len(activ)
    deltaOut = temp - actualValue
    deltatemp = np.copy(deltaOut)
    backActiv = copy(activ)
    backActiv.reverse()
    backWeight = copy(weights)
    backWeight.reverse()
    deltaWeights = []
    for x in range(len(backWeight)):
        deltatemp = np.multiply(deltatemp,dsig(backActiv[x]))
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
    nModel = {}
    nModel["biases"]=biases
    nModel["weights"]=weights
    pickle.dump(nModel,open("model.txt","w"))


def cost(inputValue,actualValue):
    output = calculate(inputValue)
    temp = output - actualValue
    temp = np.power(temp,2)
    cost = np.sum(temp)
    print "Cost = ",cost
    return cost
def costMatrix(inputValue,actualValue):
    output = calculate(inputValue)
    temp = output - actualValue
    temp = np.power(temp,2)
    temp = np.sum(temp,axis=1)
    return temp
def load():
    model = pickle.load(open("model.txt","r"))
    data = pickle.load(open("data.txt","r"))
    inX = data["x"]
    inY = data["y"]
#
# model["weights"] = weights
# model["biases"] = biases
# pickle.dump(model,open("model.txt","w"))
if __name__ =="__main__":
    times = input("Times")
    for x in range(times):
        cost(inX,inY)
        train(inX,inY)
        # load()

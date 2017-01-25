import numpy as np
import pickle
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from train import calculate,costMatrix
def predict(temp):
    shape = (2,3,2)
    model = pickle.load(open("model.txt","r"))
    biases = model["biases"]
    weights = model["weights"]
    result = calculate(temp)
    return np.argmax(result,axis=1)
    # for x in range(len(shape)-1):
    #     temp = temp.dot(weights[x]) + biases[x]
    #     temp = np.tanh(temp)
    # exp_score = np.exp(temp)
    # probs = exp_score/np.sum(exp_score,axis=0)
    # return np.argmax(probs,axis=1)
def plot_cost(X,y):
    cost = costMatrix(X,y)
    xcoord, ycoord = np.hsplit(X,2)
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    ax.scatter(xcoord,ycoord,cost)
    plt.show()

def plot_decision_boundary(pred_func, X, y):
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=np.argmax(y,axis=1), cmap=plt.cm.Spectral)
    plt.show()
def show(inp):
    data = pickle.load(open("data.txt","r"))
    if inp==1:
        plot_cost(data["x"],data["y"])
    elif inp==2:
        plot_decision_boundary(lambda x : predict(x),data["x"],data["y"])
if __name__ =="__main__":
    introString=\
"""                     Neural Network Prediction Plotting Tool
                        Enter 1 for Plotting Cost vs Input
                        Enter 2 for Prediction Plot"""
    inp = input(introString)
    show(inp)

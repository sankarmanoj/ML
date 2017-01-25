import matplotlib.pyplot as plt
import tensorflow as tf
import sklearn
from sklearn import datasets
import numpy as np
plt.ion()
np.random.seed(0)
def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))
X, Y_raw = datasets.make_moons(200,noise=0.3)
Y = np.zeros(shape=(200,2))
for val in range(len(Y_raw)):
    Y[val][Y_raw[val]]=1
print Y
print X.shape
print Y.shape
learning_rate = tf.placeholder(tf.float32, shape=[])
xtf = tf.placeholder(tf.float32,[None,2])
ytf = tf.placeholder(tf.float32,[None,2])
w1 = init_weights([2,5])
w2 = init_weights([5,3])
w_output = init_weights([3,2])
b1 = tf.Variable(tf.random_normal([5]))
b2 = tf.Variable(tf.random_normal([3]))
b_output = tf.Variable(tf.random_normal([2]))


h1 = tf.nn.tanh(tf.matmul(xtf,w1)+b1)
h2 = tf.nn.tanh(tf.matmul(h1,w2)+b2)
y_calc = (tf.matmul(h2,w_output)+b_output)

predict = y_calc


cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_calc,labels=ytf))
train_op = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)
lr =2

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
for i in range(100000):
    sess.run(train_op,feed_dict={xtf:X,ytf:Y})
    if i%1000 == 0:
        print lr
        lr = lr*0.9
        # train_op = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(cost)
        # Set min and max values and give it some padding
        x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
        y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
        h = 0.01
        # Generate a grid of points with distance h between them
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        # Predict the function value for the whole gid
        Z = np.argmax(sess.run(predict,feed_dict={xtf:np.c_[xx.ravel(), yy.ravel()]}),axis=1)
        Z = Z.reshape(xx.shape)
        # Plot the contour and training examples
        plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
        plt.scatter(X[:, 0], X[:, 1], c=Y_raw, cmap=plt.cm.Spectral)
        plt.pause(0.2)

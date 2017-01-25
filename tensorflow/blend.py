import cv2
import tensorflow as tf
a = cv2.resize(cv2.imread("bob.png"),(512,512))
b = cv2.resize(cv2.imread("alice.png"),(512,512))
atf = tf.placeholder(tf.float32,shape=(512,512,3))
btf = tf.placeholder(tf.float32,shape=(512,512,3))
y = (atf+btf)/2
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
cv2.imwrite("dog.png",sess.run(y,feed_dict={atf:a,btf:b}))

import numpy as np
import cv2
from os import listdir
import os
import pickle
from train import calculate
import numpy as np
from random import randint
from os.path import isfile, join
carfiles = [join("car",f) for f in listdir("car") if isfile(join("car", f))]
facefiles=[]
for root,bob,files in os.walk("face"):
    facefiles +=[join(root,x) for x in files]
del facefiles[0]
del facefiles[0]
carfiles = carfiles[0:len(facefiles)]
allfiles = carfiles + facefiles
def predict(temp):
    while True:
        num = randint(0,len(allfiles))
        img = cv2.imread(allfiles[num],0)
        img = cv2.resize(img,(128,128))
        returnType = np.argmax(calculate(img.flatten()))
        if returnType==0:
            print "Car"
            cv2.imshow("Car",img)
        elif returnType==1:
            cv2.imshow("Face",img)
            print "Face"

        cv2.waitKey(2000)


predict(2)

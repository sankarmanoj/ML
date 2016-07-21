import cv2
from os import listdir
import os
import pickle
import numpy as np
from os.path import isfile, join
carfiles = [join("car",f) for f in listdir("car") if isfile(join("car", f))]
facefiles=[]
for root,bob,files in os.walk("face"):
    facefiles +=[join(root,x) for x in files]
del facefiles[0]
del facefiles[0]
carimgs = []
for x in carfiles:
    inputfile = cv2.imread(x,0)
    inputfile = np.array(inputfile.flatten())
    print inputfile.shape
    if inputfile is not None:
        carimgs.append(inputfile)
faceimgs = []
for x in facefiles:
    inputfile = cv2.imread(x,0)
    inputfile = cv2.resize(inputfile,(128,128))
    inputfile = np.array(inputfile.flatten())
    print inputfile.shape
    if inputfile is not None:
        faceimgs.append(inputfile)
with open("faces.dat","w") as f:
    pickle.dump(np.array(faceimgs),f)
with open("cars.dat","w") as f:
    pickle.dump(np.array(carimgs),f)

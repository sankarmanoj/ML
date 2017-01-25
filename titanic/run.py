import csv
import matplotlib.pyplot as plt
import pandas as pd
ages = [0]*100
aged = [0]*100
reader = csv.reader(open("train.csv","r"))
reader.next()
for row in reader:
    try:
        if int(row[1]):
            ages[int(row[5])]+=1
        else:
            aged[int(row[5])]+=1
    except:
        print row
print ages
print aged
plt.plot(range(100),ages)
plt.plot(range(100),aged)
plt.show()

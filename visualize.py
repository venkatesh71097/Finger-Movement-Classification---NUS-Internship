import numpy as np
import pandas as pd
import os
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
#from keras import backend as K
from keras import optimizers
#K.set_image_dim_ordering('th')
# setting up a random seed for reproducibility
random_seed = 611
np.random.seed(random_seed)
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

label= np.empty((0))
r = []
q = ['t','i','r','l','s']
for k in range(5):	 
    path = "/home/venkatesh/Desktop/myonet/trainingsets/sub%d"%(k+1)
    os.chdir(path)
    for j in range(5):
        for i in range(30):
            dataset = []
            count = 0
            with open(q[j]+'%d.txt'%(i+1),'r') as f:
                lines = (f.read().splitlines())
                print(f.name)
                lines = [w.replace("[", '') for w in lines]
                lines = [w.replace(']', '') for w in lines]
                lines = [w.replace('  ', '') for w in lines]
                lines = [w.replace(' ', '') for w in lines]
                print(len(lines))
                for l in lines:
                    li = l.split(",")
                    li = [float(ele) for ele in li]
                    count = count + 1 
                    if count <= 90:
                        dataset.append(li)
                        
            a = np.asarray(dataset)
            plt.figure()
            plt.imshow(a, cmap=plt.get_cmap('gray'))
            plt.savefig('vis%d'%j + '-%d'%(i+1) + '.jpg')
            plt.show()
            label = np.append(label, j)
            print(label)
        

import numpy as np
from keras.models import load_model
import os
import tensorflow as tf
with tf.device('/gpu:0'):
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    model = load_model("/data/venki/myonet/trainingsets/sub5/model_EMG_CNN_5.h5")
    dataset = []
    i=0
    k=0
    c0 =0
    c1 =0
    c2 =0
    c3 =0
    c4 =0
    c5 =0

    q = ['t','i','m','r','l','s']
    with tf.device('/gpu:2'):
        for k in range(1):
            path = "/data/venki/myonet/trainingsets/sub%d"%(k+6)
            os.chdir(path)
            print(path)
            for j in range(6):
                for i in range(30):
                    dataset = []
                    count = 0
                    with open(q[j]+'%d.txt'%(i+1),'r') as f:
                        lines = (f.read().splitlines())
                        #print(f.name)
                        lines = [w.replace("[", '') for w in lines]
                        lines = [w.replace(']', '') for w in lines]
                        lines = [w.replace('  ', '') for w in lines]
                        lines = [w.replace(' ', '') for w in lines]
                        #print(len(lines))
                        for l in lines:
                            li = l.split(",")
                            li = [float(ele) for ele in li]
                            count = count + 1 
                            if count <= 90:
                                dataset.append(li)

                    da = np.array(dataset)
                    segments = da.reshape(1,90,8,1)
                    a = model.predict_classes(segments)
                    foo = open("test_sub6 %d.txt"%k,'a')
                    foo.write(str(a) + "\n")
                    if a == 0:
                        c0 = c0+1
                    elif a == 1:
                        c1 = c1+1
                    elif a == 2:
                        c2 = c2+1
                    elif a == 3:
                        c3 = c3+1
                    elif a == 4:
                        c4 = c4+1
                    elif a == 5:
                        c5 = c5+1

        print(c0)
        print(c1)
        print(c2)
        print(c3)
        print(c4)
        print(c5)

                    #with open('/data/venki/myonet/summa.txt','a') as file: 
                        #file.write("sample %d"%(i+1) + str(a) + '\n')
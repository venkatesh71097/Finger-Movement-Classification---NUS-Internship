import numpy as np
import pandas as pd
import os
import keras
os.environ["CUDA_VISIBLE_DEVICES"]="2"

#Keras package initializers

from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, BatchNormalization, Activation
from keras.layers.normalization import BatchNormalization
#from keras import backend as K
from keras import optimizers
import tensorflow as tf 
from keras.callbacks import ModelCheckpoint
from keras.callbacks import CSVLogger
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
import matplotlib.pyplot as plt

#logs the epochial data to CSV file 
csv_logger = CSVLogger('tr_final.csv', append=True, separator=';')

co0=0 
co1=0
co2=0
co3=0
co4=0
co5=0
c0=0 
c1=0
c2=0
c3=0
c4=0
c5=0
p = 0
class TestCallback(keras.callbacks.Callback):
    def __init__(self, test_data):
        self.test_data = test_data

    def on_epoch_end(self, epoch, logs={}):
        global co0,co1,co2,co3,co4,co5,c0,c1,c2,c3,c4,c5,p
        x, y = self.test_data
        a = model.predict_classes(x)
        for z in range(len(x)):
            q = open('est_CNN_final %d.txt'%p,'a')
            if testY[z][0] == 1: 
                co0 = co0+1
            elif testY[z][1] == 1: 
                co1 = co1+1
            elif testY[z][2] == 1: 
                co2 = co2+1
            elif testY[z][3] == 1: 
                co3 = co3+1
            if y[z][0] == 1 and a[z] == 0: 
                c0 = c0+1
                q.write("line %d has"%z + "0\n")
            elif y[z][1] == 1 and a[z] == 1: 
                c1 = c1+1
                q.write("line %d has"%z + "1\n")
            elif y[z][2] == 1 and a[z] == 2: 
                c2 = c2+1
                q.write("line %d has"%z + "2\n")
            elif y[z][3] == 1 and a[z] == 3: 
                c3 = c3+1
                q.write("line %d has"%z + "3\n")
        p = p+1
        loss, acc = self.model.evaluate(x, y, verbose=1)
        print('\nTesting loss: {}, acc: {}\n'.format(loss, acc))
        bar = open("test_epochs_CNN_final.csv",'a')
        bar.write(str(loss) + ", " + str(acc) + "\n")
        foo = open("expectation_CNN_final.txt",'a')
        foo.write(str(co0) + ", " + str(co1) + ", " + str(co2) + ", " + str(co3) + "," + str(co4) + "," + str(co5) + "\n")
        boo = open("correct_CNN_final.txt",'a')
        boo.write(str(c0) + ", " + str(c1) + ", " + str(c2) + ", " + str(c3) +  str(co4) + "," + str(co5) + "\n")

#K.set_image_dim_ordering('th')
# setting up a random seed for reproducibility
random_seed = 611
np.random.seed(random_seed)
dataset = []
label= np.empty((0))
r = []
q = ['t','i','r','l','s']
s = ['t','i','r']

#reads the data from the trainingsets file - 10 subjects
with tf.device('/gpu:2'):
    for k in range(9):	 
        path = "/data/venki/myonet/trainingsets/sub%d"%(k+1)
        os.chdir(path)
        for j in range(5):
            for i in range(20):
                if k > 6:
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
                    label = np.append(label, j)
                    print(label)
    
    #output of the dataset                 
    labels = np.asarray(pd.get_dummies(label),dtype = np.int8)
    print(len(dataset))
    #print(dataset)
    d = np.array(dataset)
    print(d.shape)
    segments = d.reshape(1000,90,8)
    numOfRows = segments.shape[1]
    numOfColumns = segments.shape[2]
    numChannels = 1
    numFilters = 256 # number of filters in Conv2D layer
    # kernal size of the Conv2D layer
    kernalSize1 = 2
    # max pooling window size
    poolingWindowSz = 2
    # number of filters in fully connected layers
    numNueronsFCL1 = 256
    numNueronsFCL2 = 256
    # split ratio for test and validation
    # number of epochs
    Epochs = 200
    # batchsize 
    batchSize = 4
    # number of total clases
    numClasses = labels.shape[1]
    # dropout ratio for dropout layer
    dropOutRatio = 0.1
    # reshaping the data for network input
    reshapedSegments = segments.reshape(segments.shape[0], numOfRows, numOfColumns,1)
    # splitting in training and testing data
    """trainSplitRatio = 0.60
    trainSplit = np.random.rand(len(reshapedSegments)) < trainSplitRatio
    trainX = reshapedSegments[trainSplit]
    testX = reshapedSegments[~trainSplit]
    trainX = np.nan_to_num(trainX)
    testX = np.nan_to_num(testX)
    trainY = labels[trainSplit]
    testY = labels[~trainSplit]
    zprint(trainX.shape)"""
    
    #splitting into training and testing datasets
    trainX,testX = reshapedSegments[:800,:],reshapedSegments[800:,:]
    trainY,testY = labels[:800,:],labels[800:,:]
    trY, teY     = label[:800],label[800:]

    #printing the shape of the trainX for verification
    
    print(trainX.shape)
    
    #initializing the CNN model! 
    def cnnModel():
        model = Sequential()
        # adding the first convolutionial layer with 32 filters and 5 by 5 kernal size, using the rectifier as the activation function
        model.add(Conv2D(filters = 8, kernel_size = (7,7),padding = 'Same', 
                         activation ='relu', input_shape = (90,8,1)))
        model.add(BatchNormalization())
        model.add(Conv2D(filters = 8, kernel_size = (7,7),padding = 'Same', 
                         activation ='relu'))
        model.add(BatchNormalization())
        model.add(MaxPool2D(pool_size=(2,2)))


        model.add(Conv2D(filters = 16, kernel_size = (5,5),padding = 'Same', 
                         activation ='relu'))
        model.add(BatchNormalization())
        model.add(Conv2D(filters = 16, kernel_size = (5,5),padding = 'Same', 
                         activation ='relu'))
        model.add(BatchNormalization())
        model.add(MaxPool2D(pool_size=(2,2)))
        #model.add(Dropout(0.25))

        model.add(Conv2D(filters = 32, kernel_size = (3,3),padding = 'Same', 
                         activation ='relu'))
        model.add(BatchNormalization())

        model.add(Conv2D(filters = 32, kernel_size = (3,3),padding = 'Same', 
                         activation ='relu'))
        model.add(BatchNormalization())

        model.add(MaxPool2D(pool_size=(1,1)))
        #model.add(Dropout(0.25))

        model.add(Conv2D(filters = 64, kernel_size = (1,1),padding = 'Same', 
                         activation ='relu'))
        model.add(BatchNormalization())

        model.add(Conv2D(filters = 64, kernel_size = (1,1),padding = 'Same', 
                         activation ='relu'))
        model.add(BatchNormalization())

        model.add(MaxPool2D(pool_size=(1,1)))
        #model.add(Dropout(0.25))
        model.add(Flatten())
        
        #Fully connected networks! 

        model.add(Dense(512, activation = "relu",use_bias = True))
        model.add(Dense(1024, activation = "relu",use_bias = True))
        model.add(Dense(2048, activation = "relu",use_bias = True))
        model.add(Dense(4096, activation = "relu",use_bias = True))

        #model.add(Dropout(0.5))


        #model.add(Dropout(0.5))
        
        #adding the final dense network 
        model.add(Dense(5, activation = "softmax"))
        #optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
        model.compile(optimizer = 'adam', loss = "categorical_crossentropy", metrics=["accuracy"])

        return model
        # Compiling the model to generate a model
        # model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model = cnnModel()
    for layer in model.layers:
        print(layer.name)
    print(len(testX))
    print(model.summary())
    learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                        patience=3, 
                                        verbose=1, 
                                        factor=0.5, 
                                        min_lr=0.00001)
    
    #data_augmentation_done
    datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False, # apply ZCA whitening
            rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
            zoom_range = 0.15, # Randomly zoom image 
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=True)  # randomly flip images
    #fitting our training data on the datagen 
    datagen.fit(trainX)
    
    history = model.fit_generator(datagen.flow(trainX,trainY, batch_size=4),
                              epochs = 200, validation_data = (testX,testY),
                              verbose = 2, steps_per_epoch=trainX.shape[0] // 4
                              , callbacks=[learning_rate_reduction])
    #model.fit(trainX,trainY,epochs=200,batch_size=4,verbose=2)
    #the evaluated value! 
    
    score = model.evaluate(testX,testY,verbose=1)
    print(score)
    
    #predicted value of the model for each testX value
    y_pred = model.predict(testX)
    a = np.argmax(y_pred,axis=1)
    print(a)
    
    #getting the confusion matrix
    from sklearn.metrics import confusion_matrix
    print(confusion_matrix(teY, a))

    print(a)
    print('Baseline Error: %.2f%%' %(100-score[1]*100))
    model.save('model_EMG_cnnnet.h5')
    np.save('groundTruth_cnnnet.npy',testY)
    np.save('testData_cnnnet.npy',testX)
    for z in range(len(a)):
        q = open('est_cnnnet.txt','a')
        if testY[z][0] == 1: 
            co0 = co0+1
        elif testY[z][1] == 1: 
            co1 = co1+1
        elif testY[z][2] == 1: 
            co2 = co2+1
        elif testY[z][3] == 1: 
            co3 = co3+1
        elif testY[z][4] == 1: 
            co4 = co4+1
        if testY[z][0] == 1 and a[z] == 0: 
            c0 = c0+1
            q.write("line %d has"%z + "0\n")
        elif testY[z][1] == 1 and a[z] == 1: 
            c1 = c1+1
            q.write("line %d has"%z + "1\n")
        elif testY[z][2] == 1 and a[z] == 2: 
            c2 = c2+1
            q.write("line %d has"%z + "2\n")
        elif testY[z][3] == 1 and a[z] == 3: 
            c3 = c3+1
            q.write("line %d has"%z + "3\n")
        elif testY[z][4] == 1 and a[z] == 4: 
            c4 = c4+1
            q.write("line %d has"%z + "3\n")
    foo = open("expectation_cnnnet.txt",'a')
    foo.write(str(co0) + ", " + str(co1) + ", " + str(co2) + ", " + str(co3) + ", " + str(co4) + ", " + str(co5) + "\n")
    boo = open("correct_caps_cnnnet.txt",'a')
    boo.write(str(c0) + ", " + str(c1) + ", " + str(c2) + ", " + str(c3) + ", " + str(co4) + ", " + str(co5) +  "\n")

    

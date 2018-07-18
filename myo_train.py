import time
import os
import myo
import sys
import math
import numpy as np
from pybrain.datasets import SupervisedDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure import TanhLayer
from pybrain.structure import SoftmaxLayer
import config
import glob
from gui import *
import scipy as sp 
import scipy
from scipy import signal

from math import ceil, floor

e = []
emg_correctmean = []
emg_filtered = []
emg_rectified = []
emg_envelope = []
normalized = []
low_pass = 4
sfreq = 1000
app = myGUI()

running_list = []

last_gyro = ()

def listdir_nohidden(path):
    for f in os.listdir(path):
        if not f.startswith('.'):
            yield f



#11 items - 8 muscle impulses, 3 gyros, 3 files (3 gestures)
data = SupervisedDataSet(8, config.numGestures)

#Todo: Do for all files 

index = 0;

for file in listdir_nohidden('train'): 
	f = open(os.getcwd() + '/train/' + file, "r")
	print (f.name)

	for line in f:
		line = line.replace(" ","")
		line = line.replace("[","")
		line = line.replace("]","")
		line = line.replace("  ","")
		print (line)
		datapoints = [float(x) for x in line.split(',')]
		result = [0] * (config.numGestures)
		
		#if(index < len(result)): 
		result[index] = 1
		print (result)
		print (datapoints)
		data.addSample(datapoints, result)

	print ("next")
	index = index + 1
	time.sleep(0.1)

trndata, tstdata = data.splitWithProportion(0.8)

#set the normalization limits from 0 to 1
net = buildNetwork(trndata.indim, 500, tstdata.outdim, hiddenclass=TanhLayer, outclass=SoftmaxLayer, bias=True)
trainer = BackpropTrainer(net, trndata, learningrate=0.01, lrdecay=1, momentum=0.00, verbose=False, batchlearning=False, weightdecay=0.0)
trainer.setData(trndata)
trainer.trainUntilConvergence(verbose=True,
                              trainingData=data,
                              maxEpochs=1)

net.offset =0


m = myo.Myo()
e = []
def proc_emg(emg, moving, times=[]):

	global e,emg_correctmean,emg_filtered,emg_rectified,low_pass,sfreq,emg_envelope
	e = emg
	#emg_correctmean = e - np.mean(e)
	emg_correctmean = scipy.signal.detrend(e)
	high = 20/(1000/2)
	low = 450/(1000/2)
	b, a = sp.signal.butter(4, [high,low], btype='bandpass')
	emg_filtered = sp.signal.filtfilt(b, a, e, method = 'gust')
	emg_rectified = abs(emg_filtered)
	l = float(low_pass / (sfreq/2))
	b2, a2 = sp.signal.butter(4, l, btype='lowpass')        	
	emg_envelope =sp.signal.filtfilt(b2, a2, emg_rectified,method = 'gust')
	emg_envelope = emg_envelope 
	global running_list
	if len(emg_envelope) > 0:
		array = net.activate(emg_envelope)
		print(array)
	if(len(running_list) < 15): 
		running_list.append(find_largest(array))
	else: 
		app.setText(config.result_array[find_common(running_list)])
		running_list = []

def proc_imu(quat, accel, gyro, times=[]): 
	global last_gyro
	last_gyro = gyro
	global q,a,g,b,t,c
	q = quat 
	a = accel	
	g = gyro

m.add_imu_handler(proc_imu)
m.add_emg_handler(proc_emg)

m.connect()

def find_common(result_list): 
	count_array = [0] * config.numGestures

	for index in result_list: 
		count_array[index] = count_array[index] + 1

	return find_largest(count_array) 


def find_largest(array): 	
	highest = -1
	index = 0

	for x in range(0, len(array)):  
		if(highest < array[x]): 
			highest = array[x]
			index = x

	return index

try:
	while True:
		m.run()
		
except KeyboardInterrupt:
	pass
finally:
	m.disconnect()
	print()



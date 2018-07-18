import myo
import sys
from threading import Thread
import time
import numpy as np
import os
import matplotlib
matplotlib.use('GTKAgg')
import scipy

from math import ceil, floor

e = []
emg_correctmean = []
emg_filtered = []
emg_rectified = []
emg_envelope = []
normalized = []
import config
count =0
lastLine = ""
import scipy as sp
from scipy import signal
low_pass = 4
sfreq = 1000
#delete training data from last run
from sklearn.preprocessing import MinMaxScaler

start_time = time.time()
#exit process
def buildData(name):
	m = myo.Myo(sys.argv[1] if len(sys.argv) >= 2 else None)
	_dir = os.path.join('/home/venkatesh/Desktop', 'sub1')


	if not os.path.exists(_dir):
	    	os.makedirs(_dir)


	f = open("tra/sub1/" + name + ".txt", "a")


	#Callback for EMG data from Myo (8 words)
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
		emg_envelope = emg_envelope * 100


		#print(emg_envelope)	    

	#Callback for other motion data, including accelerometer and gycroscope 
	def proc_imu(quat, acc, gyro, times=[]):
		global q,a,g,b,t,c,count
		q = quat 
		a = acc 	
		g = gyro
		if count < config.samples:
			if len(emg_envelope) > 0:
				c = list(emg_envelope)
				print(str(c) + "\n")
				f.write(str(c) + "\n")
				#plt.plot(emg_rectified)
				count = count + 1 


	m.connect()
	m.add_emg_handler(proc_emg)
	m.add_imu_handler(proc_imu)

	for x in range(0,config.samples):
		m.run()
	m.disconnect()


for i in range(0,config.numGestures):
	thread = Thread(target = buildData, args = ("s" + str(i+9), ))
	thread.deamon = True
	thread.start()
	thread.join()

	print ("Next attempt")
	input("Press Enter to continue...")
	count = 0




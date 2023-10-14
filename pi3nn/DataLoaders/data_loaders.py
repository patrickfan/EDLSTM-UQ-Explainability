''' Data loaders for Multi-step time series forecasting '''

import os 
import numpy as np 
from numpy import split
from numpy import array
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

class CL_dataLoader:
	def __init__(self, original_data_path=None, configs=None):
		if original_data_path:
			self.data_dir = original_data_path
		if configs:
			self.configs = configs

# split a univariate dataset into train/test sets
	def split_dataset(self,data, Ntrain, nfuture):
		# split into standard weeks
		train, test = data[:Ntrain], data[Ntrain:]
		# restructure into windows of weekly data
		train = array(split(train, len(train)/nfuture))
		test = array(split(test, len(test)/nfuture))
		#print (train.shape, test.shape)
		return train, test    
    
    # convert history into inputs and outputs
	def convert_supervised(self,train, n_input, n_out=7):
		# flatten data
		data = train.reshape((train.shape[0]*train.shape[1], train.shape[2]))
		X, y = list(), list()
		in_start = 0
		# step over the entire history one time step at a time
		for _ in range(len(data)):
			# define the end of the input sequence
			in_end = in_start + n_input
			out_end = in_end + n_out
			# ensure we have enough data for this instance
			if out_end <= len(data):
				X.append(data[in_start:in_end, :-1])
				y.append(data[in_end:out_end, -1])
			# move along one time step
			in_start += 1
		#print (array(X).shape, array(y).shape)
		return np.asarray(X).astype(np.float32), np.asarray(y).astype(np.float32)

	def load_inflow_timeseries(self, configs, return_original_ydata=False):

		data_name = configs['data_name']
		logtrans = configs['ylog_trans']
		plot_origin = configs['plot_origin']
		inputs_smoothing = configs['inputs_smoothing']
		scale_down = configs['scale_d']
		scale_up   = configs['scale_u']

		rawData_np = np.loadtxt(os.path.join(self.data_dir, data_name + '.dat'), delimiter=',')

		ndays = configs['ndays']     # 30
		nfuture = configs['nfuture'] # 7 
		ninputs = configs['num_inputs'] # num_inputs 3
		Ntrain = configs['Ntrain']  # 10220


		print ("-- Data name:", data_name)
		print ("-- Raw data shape: {}".format(rawData_np.shape))
		print ("-- Num of inputs: {}".format(ninputs))

        #split into train and test
		train, test = self.split_dataset(rawData_np, Ntrain, nfuture)
		train_x, train_y = self.convert_supervised(train, ndays, nfuture)

		x_temp = train_x.reshape((train_x.shape[0],train_x.shape[1]*train_x.shape[2]))
		y_temp = train_y
		xTrain, xValid, yTrain, yValid = train_test_split(pd.DataFrame(x_temp), pd.DataFrame(y_temp), test_size=0.1, random_state=0)
		xTrain_idx = xTrain.index
		xValid_idx = xValid.index

		xTrain = xTrain.to_numpy()
		yTrain = yTrain.to_numpy()
		xValid = xValid.to_numpy()
		yValid = yValid.to_numpy()


		history = [x for x in train]
		data1 = array(history)
		data1 = data1.reshape((data1.shape[0]*data1.shape[1], data1.shape[2]))
		xTest = [data1[-ndays:, :ninputs]]
		for i in range (len(test)-1):
			history.append(test[i,:])
			data = array(history)
			data = data.reshape((data.shape[0]*data.shape[1], data.shape[2]))
			input_x = data[-ndays:, :ninputs]
			xTest.append(input_x)
		xTest = array(xTest)   
		yTest = test[:,:,-1]
		xTest = xTest.reshape((xTest.shape[0],xTest.shape[1]*xTest.shape[2]))


		print ('--- xTrain shape: {}, yTrain shape: {}'.format(xTrain.shape, yTrain.shape))
		print ('--- xValid shape: {}, yValid shape: {}'.format(xValid.shape, yValid.shape))
		print ('--- xTest shape: {}, yTest shape: {}'.format(xTest.shape, yTest.shape))
		print ('--- xTrain index shape: {}, xValid index shape: {}'.format(xTrain_idx.shape, xValid_idx.shape))

        #### to float32
		xTrain = xTrain.astype(np.float32)
		yTrain = yTrain.astype(np.float32)
		xValid = xValid.astype(np.float32)
		yValid = yValid.astype(np.float32)
		xTest = xTest.astype(np.float32)
		yTest = yTest.astype(np.float32)

        ### keep the original y data without scaling
		ori_yTrain = yTrain   
		ori_yValid = yValid
		ori_yTest = yTest

        # ---scale training data both X and y
		#scalerx = StandardScaler()
		scalerx = MinMaxScaler(feature_range=(scale_down, scale_up))
		xTrain = scalerx.fit_transform(xTrain)
		xValid = scalerx.transform(xValid)
		xTest  = scalerx.transform(xTest)

		#scalery = StandardScaler()	
		scalery = MinMaxScaler(feature_range=(scale_down, scale_up))
		yTrain = scalery.fit_transform(yTrain)
		yValid = scalery.transform(yValid)
		yTest  = scalery.transform(yTest)

        # reshape input to 3D [samples, timesteps, features]        
		xTrain = xTrain.reshape((xTrain.shape[0], ndays, ninputs))
		xValid = xValid.reshape((xValid.shape[0], ndays, ninputs))
		xTest  = xTest.reshape((xTest.shape[0], ndays, ninputs))
		# yTrain = yTrain.reshape((xTrain.shape[0], nfuture, 1))
		# yValid = yValid.reshape((xValid.shape[0], nfuture, 1))
		# yTest  = yTest.reshape((yTest.shape[0], nfuture, 1))

		print ('--- xTrain shape: {}, yTrain shape: {}'.format(xTrain.shape, yTrain.shape))
		print ('--- xValid shape: {}, yValid shape: {}'.format(xValid.shape, yValid.shape))
		print ('--- xTest shape: {}, yTest shape: {}'.format(xTest.shape, yTest.shape))
		print ('--- xTrain index shape: {}, xValid index shape: {}'.format(xTrain_idx.shape, xValid_idx.shape))

		if return_original_ydata:
			return xTrain, xValid, xTest, yTrain, yValid, yTest, ninputs, nfuture, scalerx, scalery, ori_yTrain, ori_yValid, ori_yTest, xTrain_idx, xValid_idx
		else:
			return xTrain, xValid, xTest, yTrain, yValid, yTest, ninputs, nfuture, scalerx, scalery


    
    

	def standardizer(self, input_np):
		input_mean = input_np.mean(axis=0, keepdims=1)
		input_std = input_np.std(axis=0, keepdims=1)
		input_std[input_std < 1e-10] = 1.0
		standardized_input = (input_np - input_mean) / input_std
		return standardized_input, input_mean, input_std

	def getNumInputsOutputs(self, inputsOutputs_np):
		if len(inputsOutputs_np.shape) == 1:
			numInputsOutputs = 1
		if len(inputsOutputs_np.shape) > 1:
			numInputsOutputs = inputsOutputs_np.shape[1]
		return numInputsOutputs

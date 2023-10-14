import numpy as np
from math import sqrt
from numpy import split
from numpy import array
import random as rn
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from matplotlib import pyplot
from keras import Model
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import LSTM
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.layers import Input
from keras.layers import Reshape

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()  
import shap


def split_dataset(data, Ntrain):
	train, test = data[:Ntrain], data[Ntrain:]
	train = array(split(train, len(train)/7))
	test = array(split(test, len(test)/7))
	return train, test    

def convert_supervised(train, n_input, n_out=7):

	data = train.reshape((train.shape[0]*train.shape[1], train.shape[2]))
	X, y = list(), list()
	in_start = 0

	for _ in range(len(data)):
		in_end = in_start + n_input
		out_end = in_end + n_out
		if out_end <= len(data):
			X.append(data[in_start:in_end, :-1])
			y.append(data[in_end:out_end, -1])
		in_start += 1
	return np.asarray(X).astype(np.float32), np.asarray(y).astype(np.float32)

def load_inflow_timeseries():

		rawData_np = np.loadtxt(os.path.join(data_dir, data_name + '.dat'), delimiter=',')

		print ("-- Data name:", data_name)
		print ("-- Raw data shape: {}".format(rawData_np.shape))
		print ("-- Num of inputs: {}".format(ninputs))

		train, test = split_dataset(rawData_np, Ntrain)
		xTrain, yTrain = convert_supervised(train, ndays, nfuture)

		xTrain = xTrain.reshape((xTrain.shape[0],xTrain.shape[1]*xTrain.shape[2]))

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
		print ('--- xTest shape: {}, yTest shape: {}'.format(xTest.shape, yTest.shape))

		xTrain = xTrain.astype(np.float32)
		yTrain = yTrain.astype(np.float32)
		xTest = xTest.astype(np.float32)
		yTest = yTest.astype(np.float32)

		ori_yTrain = yTrain   
		ori_yTest = yTest

		scalerx = MinMaxScaler(feature_range=(0, 1))
		xTrain = scalerx.fit_transform(xTrain)
		xTest  = scalerx.transform(xTest)

		scalery = MinMaxScaler(feature_range=(0,1))
		yTrain = scalery.fit_transform(yTrain)
		yTest  = scalery.transform(yTest)
     
		xTrain = xTrain.reshape((xTrain.shape[0], ndays, ninputs))
		xTest  = xTest.reshape((xTest.shape[0], ndays, ninputs))
		yTrain = yTrain.reshape((xTrain.shape[0], nfuture, 1))
		# yTest  = yTest.reshape((yTest.shape[0], nfuture, 1))

		print ('--- xTrain shape: {}, yTrain shape: {}'.format(xTrain.shape, yTrain.shape))
		print ('--- xTest shape: {}, yTest shape: {}'.format(xTest.shape, yTest.shape))

		return xTrain, xTest, yTrain, yTest, ninputs, nfuture, scalerx, scalery, ori_yTrain, ori_yTest

data_dir  = "datasets/Timeseries/Reservoir_InflowData/"
data_name = 'CRY'

ndays = 30
nfuture =  7 
ninputs =  3
Ntrain = 8792

xTrain, xTest, yTrain, yTest, ninputs, nfuture, scalerx, scalery, ori_yTrain, ori_yTest = load_inflow_timeseries()


opt = tf.keras.optimizers.Adam(learning_rate=0.002)

def build_model(xTrain, yTrain):

	n_timesteps, n_features, n_outputs = xTrain.shape[1], xTrain.shape[2], yTrain.shape[1]

	model = Sequential()
	model.add(LSTM(200, activation='relu', input_shape=(n_timesteps, n_features)))
	model.add(RepeatVector(n_outputs))
	model.add(LSTM(200, activation='relu', return_sequences=True))
	model.add(TimeDistributed(Dense(100, activation='relu')))
	model.add(TimeDistributed(Dense(1)))
	model.compile(loss='mse', optimizer=opt)
	model.summary()
	return model

verbose = 1
epochs  = 1
batch_size = 64

model1 = build_model(xTrain, yTrain)

model1.fit(xTrain, yTrain, epochs=epochs, batch_size=batch_size, verbose=verbose)
model1.save_weights(filepath='final_weight.h5')



Day_shap = 0

def SHAP_model (xTrain, yTrain):
	n_timesteps, n_features, n_outputs = xTrain.shape[1], xTrain.shape[2], yTrain.shape[1]
	LSTM_inputs = Input(shape=(xTrain.shape[1], xTrain.shape[2]))
	x = LSTM(200, activation='relu', input_shape=(n_timesteps, n_features))(LSTM_inputs)
	x = RepeatVector(n_outputs) (x)
	x = LSTM(200, activation='relu', return_sequences=True) (x)
	x = TimeDistributed(Dense(100, activation='relu')) (x)
	x = TimeDistributed(Dense(1)) (x)
	LSTM_outputs = x[:,Day_shap,:]
	model2= Model(LSTM_inputs, LSTM_outputs, name="SHAP_model")
	model2.compile (loss='mse', optimizer=opt)
	return model2

Day_model = SHAP_model (xTrain, yTrain)
Day_model.load_weights("final_weight.h5")


#SHAP

explainer = shap.DeepExplainer(Day_model, xTrain)
shap_values = explainer.shap_values(xTest)
nobs = ndays * ninputs
shap_values = np.reshape(shap_values, (xTest.shape[0],nobs))
np.savetxt(data_name+'_Day_{}_'.format(Day_shap)+'shap_values_2D.out',shap_values)

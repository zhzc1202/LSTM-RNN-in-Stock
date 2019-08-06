#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 11:08:37 2019

@author: jason
"""

#First part
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

dataset = pd.read_csv('GOOGL.csv')
train_set = dataset.iloc[:, 1:2].values

scaler = MinMaxScaler(feature_range = (0, 1))
train_set_scaled = scaler.fit_transform(train_set)

Xtrain = train_set_scaled[0:len(train_set_scaled)-1]
Ytrain = train_set_scaled[1:len(train_set_scaled)]

Xtrain = np.reshape(Xtrain, (Xtrain.shape[0], Xtrain.shape[1], 1))

#Second part
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

model = Sequential()
model.add(LSTM(units = 4, return_sequences = True,
               activation = 'sigmoid', input_shape= (None, 1)))

model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))

model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))

model.add(LSTM(units = 50))
model.add(Dropout(0.2))

model.add(Dense(units = 1))
model.compile(optimizer = 'adam', loss = 'mean_squared_error')
model.fit(Xtrain, Ytrain, epochs = 10, batch_size = 32)
#迭代次数

#Third part
test_csv = pd.read_csv('AMD.csv')
test = test_csv.iloc[:,1:2].values

test_set_scaled = scaler.fit_transform(test)

Xtest = test_set_scaled[0:len(test)]
Xtest = np.reshape(Xtest, (Xtest.shape[0], Xtest.shape[1], 1))

predS = model.predict(Xtest)
pred = scaler.inverse_transform(predS)

plt.plot(test, color = 'red', label = 'Real price')
plt.plot(pred, color = 'blue', label = 'Predicted price')
plt.title('Stock Price Prediction')
plt.xlabel('time')
plt.ylabel('Stock price')
plt.legend()
plt.show()

#Forth part
import plotly.plotly as py
from plotly.graph_objs import Scatter, Figure, Layout
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

date = test_csv.iloc[:,0:1].values
xplot = list(np.squeeze(date))
yrplot = np.squeeze(test)
ypplot = np.squeeze(pred)


trace1 = Scatter(
        x = xplot,
        y = yrplot,
        name = 'Real stock price'
        )

trace2 = Scatter(
        x = xplot,
        y = ypplot,
        name = 'Predicted stock price'
        )

plot(
     {
      'data':[trace1, trace2],
      'layout':{
              'title' : 'visual results',
               'font' : dict(size = 16)
              }
      }
      )
'''
#RMSE
import math
from sklearn.metrics import mean_squared_error

error = math.sqrt(mean_squared_error(Xtest,pred))
error = error(sum(Xtest)/Xtest.shape[0])
percent = error * 100

print('------------------------------------------')
print('RMSE'+str(error))
print('Percentage:'+str(percent[0])+' %')
print('------------------------------------------')
'''
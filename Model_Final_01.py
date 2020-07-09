# -*- coding: utf-8 -*-
"""
Created on Tue May 15 00:11:41 2018

@author: bipra
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Feb 10 13:01:52 2018

@author: bipra
"""





# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#********************************TRAINING SET**************************

# Importing the training set
dataset_train = pd.read_csv('UNSW_NB15_training-set_5000.csv')
#training_set = dataset_train.iloc[:, 5:6].values
training_set = dataset_train.iloc[:, [3,7,10,27,35]].values

#Features:   sbytes, sttl, smean, ct_dst_sport_ltm
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
training_set[:, 0] = labelencoder_X.fit_transform(training_set[:, 0])
 
# Feature Scaling | Normalizing ---- TRAINING SET
from sklearn.preprocessing import MinMaxScaler
sclr = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sclr.fit_transform(training_set)

# Creating a data structure with 80 timesteps and 1 output
X_train = []
y_train = []
for i in range(1, 5451):
    X_train.append(training_set_scaled[i-1:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))





# Part 2 - Building the RNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import TimeDistributed
from keras.layers import Bidirectional
from numpy import array
from random import random
from numpy import cumsum
from keras import optimizers
import math


# Initialising the BLSTM
regressor = Sequential()


# Adding the first BLSTM layer and some Dropout regularisation
regressor.add(Bidirectional(LSTM(units = 180, return_sequences=True), input_shape=(X_train.shape[1], 1), merge_mode='concat'))
regressor.add(Dropout(0.2))


# Adding hidden layer
regressor.add(Bidirectional(LSTM(units = 220, return_sequences=True)))
regressor.add(Dropout(0.2))


regressor.add(Bidirectional(LSTM(units = 240, return_sequences=True)))
regressor.add(Dropout(0.2))


regressor.add(Bidirectional(LSTM(units = 260)))
regressor.add(Dropout(0.2))


# Adding the output layer
regressor.add(Dense(units = 1, activation = 'sigmoid'))


# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics=['accuracy','mse'])

# Fitting the BLSTM to the Training set | **** Splitting Training set into TRAIN & VALIDATION set ******
history = regressor.fit(X_train, y_train, epochs = 100, validation_split=0.33, batch_size = 132)

#********************************************************
x_pred = regressor.predict(X_train) #predict

x_pred = (x_pred>0.5).astype(int) #continious to integer casting
y_train = (y_train>0.5).astype(int) ##continious to integer casting


#confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_train, x_pred)

#plotting Confusion Matrix
plt.clf()
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Wistia)
classNames = ['Negative','Positive']
plt.title('Confusion Matrix - Training Data')
plt.ylabel('True label')
plt.xlabel('Predicted label')
tick_marks = np.arange(len(classNames))
plt.xticks(tick_marks, classNames, rotation=45)
plt.yticks(tick_marks, classNames)
s = [['TN','FP'], ['FN', 'TP']]
for i in range(2):
    for j in range(2):
        plt.text(j,i, str(s[i][j])+" = "+str(cm[i][j]))
plt.show()




from sklearn.metrics import classification_report
report = classification_report(y_train, x_pred)
print(report)

print(history.history['acc'])

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()


plt.plot(history.history['acc'])

plt.title('Model Accuracy')
plt.ylabel('percentage')
plt.xlabel('epoch')
plt.legend(['accuracy'], loc='upper left')
plt.show()



##************ TESTING MSE of the Network**********************************


##***********MSE becoming zero means your expected neuron outputs are exactly matched by 
#actual neuron outputs. That means network is ideally trained and goal is 100%.
#Practically speaking if you are dealing with certain classification task, 
#this means your features are enough strong to result in 100% classification rate.

#*****************************TEST SET***********************************

#Importing Test set
#dataset_test = pd.read_csv('testset_1_nor2393_att1288.csv')

#dataset_test = pd.read_csv('UNSW_NB15_testing-set_10000.csv')

dataset_test = pd.read_csv('UNSW_NB15_testing-set.csv')
test_set = dataset_test.iloc[:, [3,7,10,27,35]].values

#Features:   service, sbytes, sttl, smean, ct_dst_sport_ltm

labelencoder_X_test = LabelEncoder()
test_set[:, 0] = labelencoder_X_test.fit_transform(test_set[:, 0])
 
# Feature Scaling | Normalizing ---- TEST SET
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
test_set_scaled = sclr.fit_transform(test_set)

# Creating a data structure with 60 timesteps and 1 output
X_test = []
y_test = []
for i in range(1, 175341):
    X_test.append(test_set_scaled[i-1:i, 0])
    y_test.append(test_set_scaled[i, 0])
X_test, y_test = np.array(X_test), np.array(y_test)


# Reshaping
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))




# Performance Analysis *********************************************

# Performance Analysis *********************************************

y_pred = regressor.predict(X_test) #predict

y_pred = (y_pred>0.5).astype(int) #continious to integer casting
y_test = (y_test>0.5).astype(int) ##continious to integer casting


#confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#plotting Confusion Matrix
plt.clf()
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Wistia)
classNames = ['Negative','Positive']
plt.title('Attack or Normal Confusion Matrix - Test Data')
plt.ylabel('True label')
plt.xlabel('Predicted label')
tick_marks = np.arange(len(classNames))
plt.xticks(tick_marks, classNames, rotation=45)
plt.yticks(tick_marks, classNames)
s = [['TN','FP'], ['FN', 'TP']]
for i in range(2):
    for j in range(2):
        plt.text(j,i, str(s[i][j])+" = "+str(cm[i][j]))
plt.show()


#accuracy score
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred, normalize=False)


#classification report
from sklearn.metrics import classification_report
report = classification_report(y_test, y_pred)
print(report)



































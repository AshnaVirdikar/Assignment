# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 18:59:42 2020

@author: Ashna
"""

# K-Nearest Neighbors (K-NN)

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:,1 :-1].values
y = dataset.iloc[:, -1].values

# Taking care of missing data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X[:, 0:3])
X[:, 0:3] = imputer.transform(X[:, 0:3])


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

X_train_scaled = scaler.fit_transform(X_train)
X_train = pd.DataFrame(X_train_scaled)

X_test_scaled = scaler.fit_transform(X_test)
X_test = pd.DataFrame(X_test_scaled)

#Error rate for different K values
#import required packages
from sklearn import neighbors
from sklearn.metrics import mean_squared_error 
from math import sqrt
import matplotlib.pyplot as plt
%matplotlib inline

rmse_val = [] #to store rmse values for different k
for K in range(20):
    K = K+1
    model = neighbors.KNeighborsRegressor(n_neighbors = K)

    model.fit(X_train, y_train)  #fit the model
    pred=model.predict(X_test) #make prediction on test set
    error = sqrt(mean_squared_error(y_test,pred)) #calculate rmse
    rmse_val.append(error) #store rmse values
    print('RMSE value for k= ' , K , 'is:', error)

 #plotting the rmse values against k values (validation curve)
curve = pd.DataFrame(rmse_val) #elbow curve 
curve.plot()
 
#Implementing GridsearchCV 
from sklearn.model_selection import GridSearchCV
params = {'n_neighbors':[5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]}

knn = neighbors.KNeighborsRegressor()

model = GridSearchCV(knn, params, cv=5)
model.fit(X_train,y_train)
model.best_params_

# Training the K-NN model on the Training set
from sklearn.neighbors import KNeighborsRegressor   
regressor = KNeighborsRegressor( n_neighbors = 5 , weights='distance', p=2,metric='minkowski')
regressor.fit(X_train,y_train)
curve.plot()

#plot training error
#import required packages
from sklearn import neighbors
from sklearn.metrics import mean_squared_error 
from math import sqrt
import matplotlib.pyplot as plt
%matplotlib inline

train_rmse_val = [] #to store rmse values for different k of training set
for K in range(20):
    K = K+1
    model = neighbors.KNeighborsRegressor(n_neighbors = K)

    model.fit(X_train, y_train)  #fit the model
    z_pred=model.predict(X_test) #make prediction on test set
    training_error = sqrt(mean_squared_error(y_train[:24],z_pred)) #calculate rmse
    train_rmse_val.append(training_error) #store rmse values
    print('RMSE value for k= ' , K , 'is:', training_error)


curve = pd.DataFrame(train_rmse_val) #error curve for training set
curve.plot()
 
# Checking the accuracy
from sklearn.metrics import r2_score
print(r2_score((y_test), pred))





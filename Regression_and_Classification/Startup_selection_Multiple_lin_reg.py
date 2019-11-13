# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 14:54:46 2019

@author: sanjay.hegde
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import statsmodels.formula.api as sm

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

#encoding categorical data
labelencoder = LabelEncoder()
x[:,3] = labelencoder.fit_transform(x[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
x = onehotencoder.fit_transform(x).toarray()

# Splitting the dataset into the Training set and Test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 0)

#fitting the model to data
Reg=LinearRegression()
Reg.fit(x_train,y_train)

#predict the results
result=Reg.predict(x_test)
metrics.r2_score(y_test,result)

#optimal model using backward elimination
x=np.append(arr=np.ones((50,1)).astype(int),values=x,axis=1)
#after removing all with highest p-value
x_opt=x[:,[0,1,2,3,4]]
Reg_ols=sm.OLS(endog=y,exog=x_opt).fit() 
#ols-ordinary least square. It takes dependent and independent variable as input
Reg_ols.summary()
#the independent variable that predicts the output properly id R&D 
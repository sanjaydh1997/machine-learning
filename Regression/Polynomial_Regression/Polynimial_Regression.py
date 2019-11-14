# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 10:59:21 2019

@author: sanjay.hegde
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

regression=LinearRegression()
regression.fit(x,y)

#poly.Features will create data by raising the power of variable.
#fitting polynomial model. Try using diferent degrees.
poly=PolynomialFeatures(degree=4)
x_poly=poly.fit_transform(x)
#to make polynomial model into a multiple lin. regression model
regression_2=LinearRegression()
regression_2.fit(x_poly,y)

#visualize the model results. For Lin_model
plt.scatter(x,y,color='red')
plt.plot(x,regression.predict(x),color='green')
plt.title('True or Bluff (Comparing results of model)')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()

#the comparethe graphs of models 

#visualize the model results. For poly_model
#to make the curve better, split the x_axis intpo decimal places
plt.scatter(x,y,color='red')
plt.plot(x,regression_2.predict(poly.fit_transform(x)),color='green')
plt.title('True or Bluff (Comparing results of model)')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()

regression.predict([[6.5]])

regression_2.predict(poly.fit_transform([[6.5]]))
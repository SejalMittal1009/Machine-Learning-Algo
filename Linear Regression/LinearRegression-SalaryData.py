# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 21:21:54 2019

@author: ARYAN
"""
#import model
import numpy as np
import pandas as pd
#import module to calculate model performance
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics


def rmse(testing, prediction):
    return np.sqrt(((testing - prediction)**2).mean())

datapath='E:\\ARYAN\\Documents\\Python Scripts\\Salary_Data.csv'

data=pd.read_csv(datapath) #read csv file
#feature=['YearsExperience']
x=data[['YearsExperience']]
y=data.Salary

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1)

linreg=LinearRegression()

linreg.fit(x_train,y_train)

y_pred=linreg.predict(x_test)

print("Using own RMSE Formula: ",rmse(y_test,y_pred)) #function developed to calculate rmse value

print("Using inbuilt Formula: ",np.sqrt(metrics.mean_squared_error(y_test,y_pred))) #predefined rmse function

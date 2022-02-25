# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 21:46:52 2021

@author: uday

@Topic: Multivariate LR
"""

import pandas as pd
from sklearn import linear_model
import math


###### Data Preprocessing #############

df = pd.read_csv("homeprices.csv")

#df.bedrooms.median()
median_bedrooms = math.floor(df.bedrooms.median())

df.bedrooms = df.bedrooms.fillna(median_bedrooms)

#print(df.bedrooms)


#print(median_bedrooms)


################ Processing Done #############

############### Training ####################


reg = linear_model.LinearRegression()

reg.fit(df[['area', 'bedrooms', 'age']], df.price) #This data frame is of the form y = m1x1+m2x2+m3x3 + b

#print(reg.coef_)

#print(reg.intercept_)

################# Training Done #################


############ Prediction ############

pdf = pd.read_csv("predictinput.csv")

predict = reg.predict(pdf) ######## Prediction Done ##########

######## Importing results to csv file ##########

pdf['PredictedPrices'] = predict

pdf.to_csv("PredictionOut.csv", index=False)


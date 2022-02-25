# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 00:21:39 2021

@author: uday

Car Price Prediction
"""

import pandas as pd
from sklearn.linear_model import LinearRegression

df = pd.read_csv("carprices.csv")

dummies = pd.get_dummies(df.CarModel)

merge_df = pd.concat([df,dummies],axis="columns")

# print(merge_df)

final_df = merge_df.drop(['CarModel', 'Mercedez Benz C class'], axis="columns")

X = final_df.drop(['SellPrice($)'], axis="columns")

Y = final_df[["SellPrice($)"]]

################# Data Preprocessing is done ##################

reg = LinearRegression()

reg.fit(X,Y)

################# Training is done ##################

#Accuracy = reg.score(X,Y) * 100

print(f'Accuracy is {reg.score(X,Y) * 100} %')

print(f'{reg.predict([[45000,4,0,0]])} is the price of Mercedez with 4yr old,45000 mileage')

print(f'{reg.predict([[86000,7,0,1]])} is the price of BMW with 7yr old,86000 mileage')

# print(X)
# print("Y:-", Y)

# print(final_df)
# # print(dummies)
# print(df)
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 11:41:47 2021

@author: uday

Predicting Canada's PerCapitaIncome'
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model

indf = pd.read_csv("modelInputDataYearVSPerCapitaIncome.csv")
reg = linear_model.LinearRegression()
reg.fit(indf[['Year']], indf.PerCapitaIncome)

plt.xlabel('Year')
plt.ylabel('Per Capita')
plt.title('Canada Percapita Income')
plt.scatter(indf.Year, indf.PerCapitaIncome,color='b', marker='*')
plt.plot(indf.Year, reg.predict(indf[['Year']]), color='g')

print(reg.predict([[2020]]))



PIn = pd.read_csv("predictInput.csv")

predict = reg.predict(PIn)

PIn['EstimatedPerCapita'] = predict

PIn.to_csv("EstimatedOut.csv",index=False)

plt.xlabel('Year')
plt.ylabel('Estimated Per Capita')
plt.title('Canada Percapita Income EST')
plt.scatter(PIn.Year, PIn.EstimatedPerCapita,color='r', marker='x')
plt.plot(PIn.Year, PIn.EstimatedPerCapita, color='g')
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 10:30:01 2021

@author: udaya
"""

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("datamileage.csv")

# plot1=plt.figure(1)
# plt.scatter(df['Mileage'], df['SellPrice($)'])


# plot2=plt.figure(2)
# plt.scatter(df['Age(yrs)'], df['SellPrice($)'])

X = df[['Mileage', 'Age(yrs)']]
Y = df['SellPrice($)']


##################### Preprocess done #############

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2, random_state=10)  #0.2 means test data set as 20%, so 80% of data is trained.

#print(X_train) #The samples are very random while split to test and train. if we use ,random_state=10 in the above command then the selection is stable


from sklearn.linear_model import LinearRegression

regSplit = LinearRegression()
regSplit.fit(X_train,Y_train)


print(f'Predicted Out: {regSplit.predict(X_test)}')

print(f'Actual Out: {Y_test}')


print(f'Accuracy of the moderl: {regSplit.score(X_test, Y_test) * 100} %')



#print(df)

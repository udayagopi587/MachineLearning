# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 12:36:08 2021

@author: uday

Binary Classification, Logistic Regression
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("insurance_data.csv")

# plt.scatter(df.age, df.bought_insurance, marker='+', color='r')

X_train, X_test, y_train, y_test = train_test_split(df[['age']], df.bought_insurance, test_size = 0.1 )


model = LogisticRegression()

model.fit(X_train, y_train)


print( f'For {X_test}\ninsurance prediction is as followes {model.predict(X_test)}')

print(f'Accuracy of the model: {model.score(X_test, y_test) * 100} %')


################### We can also find the probability of buying the insurance for various ages
# print(df)


print(model.predict_proba(X_test))
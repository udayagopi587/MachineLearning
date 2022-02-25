# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 14:56:41 2021

@author: udaya
"""

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("hrdata.csv")
df.head()
left = df[df.left == 1]
left.shape
notLeft = df[df.left == 0]
notLeft.shape
df.groupby('left').mean()
pd.crosstab(df.salary,df.left).plot(kind='bar')
pd.crosstab(df.Department,df.left).plot(kind='bar')
procdf = df[['satisfaction_level', 'average_montly_hours', 'Work_accident', 'promotion_last_5years', 'salary' ]] 
procdf.head()
dummies_salary = pd.get_dummies(procdf.salary)
dummies_salary.head()
df_data_dummies = pd.concat([procdf, dummies_salary], axis='columns')
df_data_dummies

df_data_dummies = df_data_dummies.drop('salary', axis='columns')
# df_data_dummies = df_data_dummies.drop('medium', axis='columns')
print(df_data_dummies.head())

X = df_data_dummies


y = df.left

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.3)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()

model.fit(X_train, y_train)
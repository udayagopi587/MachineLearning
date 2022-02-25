# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 17:41:38 2021

@author: uday
"""

import pandas as pd
from sklearn.linear_model import LinearRegression

df = pd.read_csv("homeprices.csv")

############################################### Using dummies Pandas ############################################################

dummies = pd.get_dummies(df.town) #Created the dummy variables. Just execute till this and observe
#print(dummies)

merged = pd.concat([df,dummies],axis='columns') #pd.concat will merge two data frames.

print(merged)

#So, now we got the dummies so we need to drop the town column from the original data set.

#After that we need to drop one of the colums from dummies, because we will hit with dummies trap. So we need to drop 1 column from n dummy columns

final = merged.drop(['town', 'west windsor'], axis='columns')

print('Final', final)

# Training

model = LinearRegression()

X = final.drop(['price'], axis='columns') #Only taking the features, by droping the target.


Y = final.price

model.fit(X,Y)

#Trained Model

print(model.predict([[3400,0,0]]))

######################### Accuracy of Predicted Model ###############

print(model.score(X,Y)) #This gives 0.9573929037221871 i.e 95% accurate

#print(df)



############################################### Using Label Encoder(Sklearn):- Its kinda messy ############################################################

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder() #this is a class

dfle = df

#print(dfle)

dfle.town = le.fit_transform(dfle.town)

X = dfle.drop(['price'], axis='columns') #.values converts a data frame to a two-D array
print(X)
Y = dfle.price

modelle = LinearRegression()

modelle.fit(X,Y)

print(modelle.predict([[3400,2]]))

#print(model.score(X,Y))

print(dfle)
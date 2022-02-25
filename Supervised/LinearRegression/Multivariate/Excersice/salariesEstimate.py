# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 00:20:39 2021

@author: uday

@Salaries Estimation
"""

import pandas as pd
import pickle
from word2number import w2n
import math
from sklearn import linear_model
#from sklearn.externals import joblib
import joblib

df = pd.read_csv("salaries.csv")

#Linear Regression will works only on numbers, so the column experience is having words as five, one, etc.. So, lets convert this to numbers.

df.experience = df.experience.fillna("zero") #Since there are few empty cells.

df.experience = df.experience.apply(w2n.word_to_num)


#Finding median of test scores column

median_test_scores = math.floor(df['test_score(out of 10)'].median())

df['test_score(out of 10)'] = df['test_score(out of 10)'].fillna(median_test_scores)


############ Data Preprocessing is done, lets go with training the model ##############

reg = linear_model.LinearRegression()

reg.fit(df[['experience','test_score(out of 10)','interview_score(out of 10)']],df['salary($)'])

########### Training done ###########

############### Predict ##############


pdf = pd.read_csv("predictinput.csv")

predict = reg.predict(pdf)

############# Prediction Done ##########


pdf['PredictedSalaries'] = predict

pdf.to_csv("PredictionOut.csv", index=False)



################ Saving the Trained Model to a file ##############

with open('model_pickle', 'wb') as f: #wb stands for write binary
    pickle.dump(reg, f) #Dumped the trained model "reg" to "model_pickle" file, it will be saved in the program saved directory.

#Lets call the trained model from pickle


with open('model_pickle', 'rb') as f: #rb = read binary
    modelP = pickle.load(f) #Load the pickle to model.
    
    
print(modelP.predict([[2, 10, 5]])) #Boom!

#Therefore, this pickle can be shred to anywhere, so if we give the predict input it suplies the output.

########### This can also be done using "sklearn", this is more efficient when we have more numpy arrays ##############
    

joblib.dump(reg, 'model_joblib')
modelJ = joblib.load('model_joblib')

print(modelJ.predict([[2, 10, 7]]))

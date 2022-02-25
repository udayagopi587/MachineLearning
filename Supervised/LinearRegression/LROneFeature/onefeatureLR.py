import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model


#All the LR models will be SCIKIT-LEARN AKA sklearn

#Lets create a data frame

df = pd.read_csv("costvsarea_house.csv") #df nominclature 'data frame'
print(df)
plt.xlabel('Area in sft')
plt.ylabel('Price in USD')
plt.title('Linear Regression with One Feature')
plt.scatter(df.area, df.prices,color='red', marker='x')

reg = linear_model.LinearRegression()
reg.fit(df[['area']], df.prices)


#print(reg.predict([[4520]])) #This gives the ans in the form of y = m*x + c; m is the slope/gradient and c is the intercept.

#Lets plot the linear eqn line on the plot.
plt.xlabel('Area in sft')
plt.ylabel('Price in USD')
plt.title('Linear Regression with One Feature')
plt.scatter(df.area, df.prices,color='red', marker='x')
plt.plot(df.area, reg.predict(df[['area']]), color='blue')


print(reg.coef_)

print(reg.intercept_)


#Lets hope that the above trained model is working good.

#Lets pass new unkown data to predict the costs for various house areas.

df_p = pd.read_csv("new_predictionInput.csv")

#print(df_p)

predict = reg.predict(df_p)

df_p['Prices'] = predict #df_p['Prices'] creates a new column in the dataframe df_p.

#print(df_p)

#Importimg the prediction output to a csv file.

df_p.to_csv("predictionOutput.csv", index=False) #This creates a csv file with df_p data in it named as predictionOutput.csv, index=False means eleminating the indices while exporting



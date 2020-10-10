import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn import linear_model

# Reading Excel file
LayerivData = pd.read_excel('layeriv_regression_data.xls')

# Define Columns 
X_data = LayerivData.drop('Y',axis='columns')
Y_data = LayerivData.Y
print(X_data.ndim)

# Split Training set and Testing set
X_train = X_data.sample(frac=0.7,random_state=4538)
X_test = X_data.drop(X_train.index)
Y_train = Y_data.sample(frac=0.7,random_state=4538)
Y_test = Y_data.drop(Y_train.index)

# Training model 
reg = linear_model.LinearRegression()
reg.fit(X_train,Y_train)

# Checking Accuracy
accuracy = reg.score(X_test,Y_test)
print( "Accuracy is "+str(accuracy))

MSE = mean_squared_error(Y_test,reg.predict(X_test))
print( "Mean squared error is "+str(MSE))

print(reg.predict([[120]]))


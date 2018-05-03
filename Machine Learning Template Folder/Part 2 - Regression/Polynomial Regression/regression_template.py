
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Fitting Linear Regression to dataset 

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,y)

#Predicting the result

#Fitting Regression Model to the dataset

#Predecting the result

y_pred = regressor.predict(6.5)

#Visualing the results

plt.scatter(X, y, color = 'red') #red points, data sets ...
plt.plot(X, regressor.predict(X), color = 'blue') #Data predicted
plt.title('Truth or bluff (Polynomial Regression) ')
plt.xlabel('Position Label')
plt.ylabel('Salary')
plt.show()

#Visualing the results (for higher resolution and smoother curve)

X_grid = np.arange(min(X),max(X),0.1)
X_grid = X_grid.reshape((len(X),1))

plt.scatter(X, y, color = 'red') #red points, data sets ...
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue') #Data predicted
plt.title('Truth or bluff (Polynomial Regression) ')
plt.xlabel('Position Label')
plt.ylabel('Salary')
plt.show()


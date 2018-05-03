
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

y_pred = lin_reg.predict([6.5]) 

#Fitting polynominal Regression to the dataset

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=13)
X_poly = poly_reg.fit_transform(X)
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly,y)

#Predecting the result

y_pred2= lin_reg2.predict(poly_reg.fit_transform(6.5))

#Visualing the results
#Linear Model
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title('Truth or bluff (Linear Regression) ')
plt.xlabel('Position Label')
plt.ylabel('Salary')
plt.show()
#Polynomial Model 

X_grid = np.arange(min(X),max(X),0.1)
X_grid = X_grid.reshape((len(X_grid),1))

plt.scatter(X, y, color = 'red') #red points, data sets ...
plt.plot(X_grid, lin_reg2.predict(poly_reg.fit_transform(X_grid)), color = 'blue') #Data predicted
plt.title('Truth or bluff (Polynomial Regression) ')
plt.xlabel('Position Label')
plt.ylabel('Salary')
plt.show()


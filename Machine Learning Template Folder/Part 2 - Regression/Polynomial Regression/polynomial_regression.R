# Importing the dataset
dataset = read.csv('Position_Salaries.csv')
dataset = dataset[2:3]

#Fitting Linear Model

lin_reg = lm(formula = Salary ~ .,
             dataset)

#Fitting Polynomial Model

dataset$Level2 = dataset$Level ^ 2
dataset$Level3 = dataset$Level ^ 3
dataset$Level4 = dataset$Level ^ 4

poly_reg = lm(formula = Salary ~ .,
              data = dataset)

#Visualing Linear Regression results
library(ggplot2)
ggplot() + 
  geom_point(aes(x = dataset$Level, y = dataset$Salary),
             colour = 'red') +
  geom_line(aes(x = dataset$Level, y = predict(lin_reg, newdata = dataset)),
            colour = 'blue') + 
  ggtitle("Truth or Bluff (Linear Regression)") +
  xlab("Level") + 
  ylab("Salary")

#Visualising Polynomila Regression results 

ggplot() + 
  geom_point(aes(x = dataset$Level, y = dataset$Salary),
             colour = 'red') +
  geom_line(aes(x = dataset$Level, y = predict(poly_reg, newdata = dataset)),
            colour = 'blue') + 
  ggtitle("Truth or Bluff (Polynomial Regression)") +
  xlab("Level") + 
  ylab("Salary")


#Predecting results using Linear Model
data_test  = data.frame(Level = 6.5)
y_pred = predict(lin_reg, data_test)

#Predecting results using Polynomial Model

data_test  = data.frame(Level = 6.5,
                        Level2 = 6.5^2,
                        Level3 = 6.5^3,
                        Level4 = 6.5^4)
y_pred = predict(poly_reg, data_test)





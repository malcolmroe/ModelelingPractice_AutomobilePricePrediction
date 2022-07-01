import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

carDataFrame = pd.read_csv('automobileEDA_MD.csv')
# Set up simple linear regression model
lm = LinearRegression()
X = carDataFrame[['highway-mpg']]
Y = carDataFrame['price']
# Not input data into LinearRegression method(s)
lm.fit(X,Y)
Yhat = lm.predict(X)
print(Yhat[0:5])
print(lm.intercept_) # Prints the intercept
print(lm.coef_) # Prints the coefficient for the predictor variable
lm1 = LinearRegression()
lm1.fit(carDataFrame[['engine-size']],carDataFrame[['price']])
print(lm1.coef_)
print(lm1.intercept_)

# Building a Multiple Linear Regression Model
lm2 = LinearRegression()
Z = carDataFrame[['horsepower','curb-weight','engine-size','highway-mpg']]
lm2.fit(Z,carDataFrame['price'])
print(lm2.coef_)
print(lm2.intercept_)

# Visualizing the models to assess them
width = 12
height = 10
plt.figure(figsize=(width,height))
sns.regplot(x='highway-mpg',y='price',data=carDataFrame)
plt.ylim(0,)
plt.show()
# One thing to keep in mind when looking at a regression plot is to pay attention to how scattered the data points are around the
# regression line.This will give you a good indication of the variance of the data and whether a linear model would be the best fit or not.
# If the data is too far off from the line, this linear model might not be the best model for this data.
# Compare this to the regression plot for 'peak-rpm':
plt.figure(figsize=(width,height))
sns.regplot(x='peak-rpm',y='price',data=carDataFrame)
plt.ylim(0,)
plt.show()
# The variable "highway-mpg" has a stronger correlation with "price", it is approximate -0.704692  compared to "peak-rpm"
# which is approximate -0.101616. You can verify it using the following command:
carDataFrame[["peak-rpm","highway-mpg","price"]].corr()
# Residual Plot:
# If the points in a residual plot are randomly spread out around the x-axis, then a linear model is appropriate for the data.
plt.figure(figsize=(width,height))
sns.residplot(carDataFrame['highway-mpg'],carDataFrame['price'])
plt.show()
# We can see from this residual plot that the residuals are not randomly spread around the x-axis, leading us to believe
# that maybe a non-linear model is more appropriate for this data.


# Multiple Linear Regression
Y_hat = lm2.predict(Z)
plt.figure(figsize=(width, height))
ax1 = sns.distplot(carDataFrame['price'], hist=False, color="r", label="Actual Value")
sns.distplot(Y_hat, hist=False, color="b", label="Fitted Values" , ax=ax1)
plt.title('Actual vs Fitted Values for Price')
plt.xlabel('Price (in dollars)')
plt.ylabel('Proportion of Cars')
plt.show()
plt.close()
# We can see that the fitted values are reasonably close to the actual values since the two distributions overlap a bit.
# However, there is definitely some room for improvement. Let's see if there's a polynomial that will fit
def PlotPolly(model, independent_variable, dependent_variabble, Name):
    x_new = np.linspace(15, 55, 100)
    y_new = model(x_new)
    plt.plot(independent_variable, dependent_variabble, '.', x_new, y_new, '-')
    plt.title('Polynomial Fit with Matplotlib for Price ~ Length')
    ax = plt.gca()
    ax.set_facecolor((0.898, 0.898, 0.898))
    fig = plt.gcf()
    plt.xlabel(Name)
    plt.ylabel('Price of Cars')
    plt.show()
    plt.close()

X2 = carDataFrame['highway-mpg']
Y2 = carDataFrame['price']
f = np.polyfit(X2, Y2, 3) # This is a polynomial of the third order (cubic)
p = np.poly1d(f)
print(p)
# Call the function in order to plot
PlotPolly(p, X2, Y2, 'highway-mpg')

# Will now create a polynomial at the 11th order
f1 = np.polyfit(X2, Y2, 11)
p1 = np.poly1d(f1)
print(p1)
PlotPolly(p1,X2,Y2, 'Highway MPG')

# Perform Polynomial Transform on multiple features
pr = PolynomialFeatures(degree=2)
Z_pr = pr.fit_transform(Z)
print(Z.shape)
print(Z_pr.shape) # This has 15 more features than the original 4 in the above. Beware of overfitting when doing this


# Measure for in-sample evaluation - Calculating R^2 and MSE for all models above
print('Linear Model:')
print('The r-squared for Linear Model is: ',lm.score(X,Y))
print('We can say that ~49.66 of variation in price is due to horsepower-fit')
mse = mean_squared_error(carDataFrame['price'],Yhat)
print('The mean squared error of Price and predicted value is: ',mse)
print('Multiple Regression Model:')
print('The r-squared for Multiple Linear Model is: ',lm2.score(Z,carDataFrame['price']))
print('We can say that ~80.9356 of variation is explained by the multiple linear regression "multi-fit"')
mse2 = mean_squared_error(carDataFrame['price'],lm2.predict(Z))
print('The mean square error of price and predicted value using multifit is: ', mse2)
print('Polynomial Fit: ')
r_squared = r2_score(Y2,p(X2))
print('We can say that ~67.4195% of variation in price is explained by this polynomial fit.', r_squared)
mse3 = mean_squared_error(carDataFrame['price'],p(X2))
print('The Mean Square Error is: ',mse3)
# MRL is a better model fit in this scenario
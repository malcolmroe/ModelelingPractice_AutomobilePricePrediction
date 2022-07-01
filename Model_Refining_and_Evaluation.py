import pandas as pd
import numpy as np
from ipywidgets import interact, interactive, fixed, interact_manual
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge

# These are functions for plotting. I have no idea how they work!
def DistributionPlot(RedFunction, BlueFunction, RedName, BlueName, Title):
    width = 12
    height = 10
    plt.figure(figsize=(width, height))
    ax1 = sns.distplot(RedFunction, hist=False, color="r", label=RedName)
    ax2 = sns.distplot(BlueFunction, hist=False, color="b", label=BlueName, ax=ax1)
    plt.title(Title)
    plt.xlabel('Price (in dollars)')
    plt.ylabel('Proportion of Cars')
    plt.show()
    plt.close()
def PollyPlot(xtrain, xtest, y_train, y_test, lr, poly_transform):
    width = 12
    height = 10
    plt.figure(figsize=(width, height))
    # training data
    # testing data
    # lr:  linear regression object
    # poly_transform:  polynomial transformation object
    xmax = max([xtrain.values.max(), xtest.values.max()])
    xmin = min([xtrain.values.min(), xtest.values.min()])
    x = np.arange(xmin, xmax, 0.1)
    plt.plot(xtrain, y_train, 'ro', label='Training Data')
    plt.plot(xtest, y_test, 'go', label='Test Data')
    plt.plot(x, lr.predict(poly_transform.fit_transform(x.reshape(-1, 1))), label='Predicted Function')
    plt.ylim([-10000, 60000])
    plt.ylabel('Price')
    plt.legend()
    plt.show()

carDataframe = pd.read_csv('module_5_auto.csv')
carDataframe = carDataframe._get_numeric_data()
print(carDataframe.head())

# Split data into training and testing data
# Move target variable to new DF
y_data = carDataframe['price']
x_data = carDataframe.drop('price',axis=1)

x_train, x_test, y_train, y_test = train_test_split(x_data,y_data,test_size=0.10,random_state=1)
print('Number of test samples: ', x_test.shape[0])
print('Number of training samples: ',x_train.shape[0])

lre = LinearRegression() # Create Linear Regression Object
lre.fit(x_train[['horsepower']],y_train) # Fit the linear regression model using the training data
a = lre.score(x_train[['horsepower']],y_train) # Calculate R^2 on train data
b = lre.score(x_test[['horsepower']],y_test) # Calculate R^2 on test data
print('R^2 of Train: ',a)
print('R^2 of Test: ',b)
# Sometimes you do not have sufficient testing data; as a result, you may want to perform cross-validation.
# Let's go over several methods that you can use for cross-validation.

Rcross = cross_val_score(lre, x_data[['horsepower']],y_data, cv=4)
print(Rcross)
print('The mean of the fold is: ', Rcross.mean(), 'and the standard deviation is: ', Rcross.std())
# Using negative squared error as a score
MSE = cross_val_score(lre,x_data[['horsepower']], y_data,cv=4,scoring='neg_mean_squared_error')
print('The Means Squared Error (MSE) is: ', MSE)
# Cross_val_predict can predict the output. Splits up data into # of folds, one fold for testing, and other for train
yhat = cross_val_predict(lre, x_data[['horsepower']],y_data,cv=4)
print(yhat)

# Overfitting with Multiple Linear Regression Models
lr2 = LinearRegression()
lr2.fit(x_train[['horsepower','curb-weight','engine-size','highway-mpg']],y_train)
# Prediction using training data:
yhat_train = lr2.predict(x_train[['horsepower','curb-weight','engine-size','highway-mpg']])
print(yhat_train[0:5])
# Prediction using testing data:
yhat_test = lr2.predict(x_test[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']])
print(yhat_test[0:5])

# Model evaluation
Title = 'Distribution  Plot of  Predicted Value Using Training Data vs Training Data Distribution'
DistributionPlot(y_train, yhat_train, "Actual Values (Train)", "Predicted Values (Train)", Title)
# So far, the model seems to be doing well in learning from the training dataset. But what happens when the model encounters
# new data from the testing dataset? When the model generates new values from the test data, we see the distribution
# of the predicted values is much different from the actual target values.
Title='Distribution  Plot of  Predicted Value Using Test Data vs Data Distribution of Test Data'
DistributionPlot(y_test,yhat_test,"Actual Values (Test)","Predicted Values (Test)",Title)
# Let's create another polynomial model that might reduce some of the noise
x_train1, x_test1, y_train1, y_test1 = train_test_split(x_data, y_data, test_size=0.45, random_state=0)
pr = PolynomialFeatures(degree=5) # 5 polynomial feature
x_train_pr = pr.fit_transform(x_train1[['horsepower']])
x_test_pr = pr.fit_transform(x_test1[['horsepower']])
# Create Linear Regression model and "poly" and train it
poly = LinearRegression()
poly.fit(x_train_pr,y_train1)
yhat2 = poly.predict(x_test_pr)
print(yhat2)
# let's compare the first five values to see how they match up
print('Predicted Values: ', yhat2[0:4])
print('Actual Values: ', y_test1[0:4].values)
PollyPlot(x_train1[['horsepower']], x_test1[['horsepower']], y_train1, y_test1, poly,pr)
print(poly.score(x_train_pr,y_train1))
print(poly.score(x_test_pr,y_test1)) # This R-Squared is insanely far off, a negative R^2 is a sign of overfitting

# This function will iterate through the orders for the equation and see which one performs best
Rsqu_test = []
order = [1, 2, 3, 4]
for n in order:
    pr = PolynomialFeatures(degree=n)
    x_train_pr = pr.fit_transform(x_train1[['horsepower']])
    x_test_pr = pr.fit_transform(x_test1[['horsepower']])
    lr2.fit(x_train_pr, y_train1)
    Rsqu_test.append(lr2.score(x_test_pr, y_test1))
plt.plot(order, Rsqu_test)
plt.xlabel('order')
plt.ylabel('R^2')
plt.title('R^2 Using Test Data')
plt.text(3, 0.75, 'Maximum R^2 ')
plt.show()

# Ridge Regression
pr2 = PolynomialFeatures(degree=2)
x_train_pr2 = pr.fit_transform(x_train[['horsepower','curb-weight','engine-size','highway-mpg','normalized-losses','symboling']])
x_test_pr2 = pr.fit_transform(x_test[['horsepower','curb-weight','engine-size','highway-mpg','normalized-losses','symboling']])

RidgeModel = Ridge(alpha=1)
RidgeModel.fit(x_train_pr2, y_train)
yhat3 = RidgeModel.predict(x_test_pr2)
print('Predicted: ', yhat3[0:4])
print('Test Set: ', y_test[0:4].values)

from tqdm import tqdm

Rsqu_test = []
Rsqu_train = []
dummy1 = []
Alpha = 10 * np.array(range(0, 1000))
pbar = tqdm(Alpha)
for alpha in pbar:
    RigeModel = Ridge(alpha=alpha)
    RigeModel.fit(x_train_pr2, y_train)
    test_score, train_score = RigeModel.score(x_test_pr2, y_test), RigeModel.score(x_train_pr2, y_train)
    pbar.set_postfix({"Test Score": test_score, "Train Score": train_score})
    Rsqu_test.append(test_score)
    Rsqu_train.append(train_score)

width = 12
height = 10
plt.figure(figsize=(width, height))
plt.plot(Alpha,Rsqu_test, label='validation data  ')
plt.plot(Alpha,Rsqu_train, 'r', label='training Data ')
plt.xlabel('alpha')
plt.ylabel('R^2')
plt.legend()
plt.show()
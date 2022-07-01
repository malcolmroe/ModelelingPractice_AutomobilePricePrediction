import pandas as pd
import numpy as np
import matplotlib as plt
from matplotlib import pyplot

headers = ["symboling","normalized-losses","make","fuel-type","aspiration", "num-of-doors","body-style",
         "drive-wheels","engine-location","wheel-base", "length","width","height","curb-weight","engine-type",
         "num-of-cylinders", "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower",
         "peak-rpm","city-mpg","highway-mpg","price"]
carDataFrame = pd.read_csv('auto.csv',names=headers)
# Empty values have a "?" in this DF. Below replaces it with "Nan" or null
carDataFrame.replace("?", np.nan, inplace=True)
# print(carDataFrame.head())
# Below is a good way to check the dataset for empty values
missing_data = carDataFrame.isnull()
for column in missing_data.columns.values.tolist():
    print(column)
    print (missing_data[column].value_counts())
    print("")
# In order to deal with the missing values for these columns, replace with mean
avg_normalized_loss = carDataFrame['normalized-losses'].astype('float').mean(axis=0)
carDataFrame['normalized-losses'].replace(np.nan,avg_normalized_loss,inplace=True)
avgBore = carDataFrame['bore'].astype('float').mean(axis=0)
carDataFrame['bore'].replace(np.nan,avgBore,inplace=True)
avgStroke = carDataFrame['stroke'].astype('float').mean(axis=0)
carDataFrame['stroke'].replace(np.nan,avgStroke,inplace=True)
avgHP = carDataFrame['horsepower'].astype('float').mean(axis=0)
carDataFrame['horsepower'].replace(np.nan,avgHP,inplace=True)
avgPeakRPM = carDataFrame['peak-rpm'].astype('float').mean(axis=0)
carDataFrame['peak-rpm'].replace(np.nan,avgPeakRPM,inplace=True)

# For these columns, we will replace with the mode
print(carDataFrame['num-of-doors'].value_counts()) #.idmax() after this will return the max automatically
carDataFrame['num-of-doors'].replace(np.nan,'four',inplace=True)

# Since price is what we want to predict, we will just drop the rows that do not have price
carDataFrame.dropna(subset=['price'],axis=0,inplace=True)
# Need to reset the index, because two rows were dropped
carDataFrame.reset_index(drop=True, inplace=True)
print(carDataFrame.head())

# The next step is to check the types in the dataframe so they are the same and mallueable
print(carDataFrame.dtypes)
# Numerical values should be either floats or ints, while categories should be object
carDataFrame[["bore", "stroke"]] = carDataFrame[["bore", "stroke"]].astype("float")
carDataFrame[["normalized-losses"]] = carDataFrame[["normalized-losses"]].astype("int")
carDataFrame[["price"]] = carDataFrame[["price"]].astype("float")
carDataFrame[["peak-rpm"]] = carDataFrame[["peak-rpm"]].astype("float")
print(carDataFrame.dtypes)

# The lab is assuming that we want to change the mpg to L/100km. Shown below
carDataFrameNew = carDataFrame.copy(deep=True) # I copied the DF so that I'm not messing with it
carDataFrameNew['city-L/100km'] = 235/carDataFrame['city-mpg'] # This adds a new column
carDataFrameNew['highway-mpg'] = 235/carDataFrameNew['highway-mpg']
carDataFrameNew.rename(columns={'highway-mpg':'highway-L/100km'},inplace=True) #This replaces the old column with the new variable
print(carDataFrameNew.columns)

#Data Normalization
#Scaling the variable so that the average is 0, variance is 1, or so variable goes from 0 to 1
# Normalizing the below variables so they range from 0 to 1
carDataFrameNew['length'] = carDataFrameNew['length']/carDataFrameNew['length'].max()
carDataFrameNew['width'] = carDataFrameNew['width']/carDataFrameNew['width'].max()
carDataFrameNew['height'] = carDataFrameNew['height']/carDataFrameNew['height'].max()
print(carDataFrameNew[['length','width','height']].head())

# Binning
# Transforms the horsepower column into three categories
# Converts column to int
carDataFrameNew['horsepower'] = carDataFrameNew['horsepower'].astype(int, copy=True)
# Plot horsepower to see what the distribution looks like
# plt.pyplot.hist(carDataFrameNew['horsepower'])
# pyplot.xlabel('Horsepower')
# pyplot.ylabel('Count')
# pyplot.title('Horsepower Bins')
# pyplot.show()

# Want an equal size bandwidth, so use numpy linspace to evenly split the data up
# Since there are 3 bins of equal length, use 4 to create the dividers
bins = np.linspace(min(carDataFrameNew['horsepower']), max(carDataFrameNew['horsepower']),4)
print(bins) # This prints out the splitting points
# Set group names and use the Pandas function cut will apply the group names
groupNames = ['Low','Medium','High']
carDataFrameNew['horsepower-binned'] = pd.cut(carDataFrameNew['horsepower'], bins, labels=groupNames, include_lowest=True)
print(carDataFrameNew[['horsepower','horsepower-binned']])
print(carDataFrameNew['horsepower-binned'].value_counts())
pyplot.bar(groupNames, carDataFrameNew['horsepower-binned'].value_counts())
pyplot.xlabel('Horsepower')
pyplot.ylabel('Count')
pyplot.title('Horsepower Bins')
pyplot.show()

# Creating an indicator (dummy) variable
# Need to convert 'fuel-type' from a categorical variable to numbers for use in regression analysis
dummy_variable_1 = pd.get_dummies(carDataFrameNew['fuel-type'])
dummy_variable_1.rename(columns={'gas':'fuel-type-gas','diesel':'fuel-type-diesel'}, inplace=True)
print(dummy_variable_1.head())
carDataFrameNew = pd.concat([carDataFrameNew,dummy_variable_1],axis=1)
carDataFrameNew.drop('fuel-type', axis=1, inplace=True)
print(carDataFrameNew.head())
# Converting other categorical variables to dummies
dummy_variable_2 = pd.get_dummies(carDataFrameNew['aspiration'])
dummy_variable_2.rename(columns={'std':'aspiration-std','turbo':'aspiration-turbo'}, inplace=True)
print(dummy_variable_2)
carDataFrameNew = pd.concat([carDataFrameNew,dummy_variable_2], axis=1)
carDataFrameNew.drop('aspiration', axis=1, inplace=True)
print(carDataFrameNew.head())
# Save the new cleansed dataframe to a new CSV
carDataFrameNew.to_csv('clean_CarDataFrame.csv')
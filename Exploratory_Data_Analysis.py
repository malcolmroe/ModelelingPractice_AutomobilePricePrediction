import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

carDataframe = pd.read_csv('automobileEDA.csv')
print(carDataframe.head())
# Understand what variable we are dealing with, and select visualization method
print(carDataframe.dtypes)
# Finding the correlation between the below columns
print(carDataframe[['bore','stroke','compression-ratio','horsepower']].corr())

# Engine size as potential predictor variable of price
sns.regplot(x='engine-size', y='price', data=carDataframe)
plt.ylim(0,)
print(carDataframe[['engine-size', 'price']].corr())
plt.show()
sns.regplot(x='highway-mpg',y='price', data=carDataframe)
plt.ylim(0,)
print(carDataframe[['highway-mpg','price']].corr())
plt.show()
# Example of weak linear relationship
sns.regplot(x='peak-rpm', y='price', data=carDataframe)
print(carDataframe[['peak-rpm','price']].corr())
plt.show()

# Categorical Variables

sns.boxplot(x='body-style',y='price', data=carDataframe)
sns.boxplot(x='engine-location', y='price', data=carDataframe)
sns.boxplot(x='drive-wheels', y='price', data=carDataframe)
plt.show()

# Engine Location might be a good indicator, because of the disparate boxes. Let's investigate further
engine_loc_counts = carDataframe['engine-location'].value_counts().to_frame()
engine_loc_counts.rename(columns={'engine-location':'value_counts'}, inplace=True)
engine_loc_counts.index.name = 'engine-location'
print(engine_loc_counts.head(10))
# After viewing that engine-location only has 3 values for rear, this would not be a good indicator for price because of how low the count is

# Grouping
print(carDataframe['drive-wheels'].unique()) # Print the specific unique values in the column
df_group_one = carDataframe[['drive-wheels','body-style','price']] # Creates a variable and grabs these three rows from the df
df_group_one = df_group_one.groupby(['drive-wheels'], as_index=False).mean() # groups the drive wheels column by the values, then calculates the mean for each
print(df_group_one)
# Can also group by multiple variables:
df_gptest = carDataframe[['drive-wheels','body-style','price']]
groupedTest1 = df_gptest.groupby(['drive-wheels', 'body-style'],as_index=False).mean()
print(groupedTest1)
# Easier to visualize in a pivot table:
grouped_pivot = groupedTest1.pivot(index='drive-wheels',columns='body-style')
print(grouped_pivot)

#Heatmap time!
fig, ax = plt.subplots()
im = ax.pcolor(grouped_pivot, cmap='RdBu')
#label names
row_labels = grouped_pivot.columns.levels[1]
col_labels = grouped_pivot.index
#move ticks and labels to the center
ax.set_xticks(np.arange(grouped_pivot.shape[1]) + 0.5, minor=False)
ax.set_yticks(np.arange(grouped_pivot.shape[0]) + 0.5, minor=False)
#insert labels
ax.set_xticklabels(row_labels, minor=False)
ax.set_yticklabels(col_labels, minor=False)
#rotate label if too long
plt.xticks(rotation=90)
fig.colorbar(im)
plt.show()

# Correlation and Causation
pearson_coef, p_value = stats.pearsonr(carDataframe['wheel-base'],carDataframe['price'])
print("The Pearson Coefficient is ", pearson_coef, " with a p-value of P = ", p_value)
# Conclusion here - correlation is statistically significant, but linear relationship isn't strong
pearson_coef, p_value = stats.pearsonr(carDataframe['horsepower'],carDataframe['price'])
print("The Pearson Coefficient is ", pearson_coef, " with a p-value of P = ", p_value)
pearson_coef, p_value = stats.pearsonr(carDataframe['length'],carDataframe['price'])
print("The Pearson Coefficient is ", pearson_coef, " with a p-value of P = ", p_value)
pearson_coef, p_value = stats.pearsonr(carDataframe['width'],carDataframe['price'])
print("The Pearson Coefficient is ", pearson_coef, " with a p-value of P = ", p_value)
pearson_coef, p_value = stats.pearsonr(carDataframe['curb-weight'],carDataframe['price'])
print("The Pearson Coefficient is ", pearson_coef, " with a p-value of P = ", p_value)
pearson_coef, p_value = stats.pearsonr(carDataframe['engine-size'],carDataframe['price'])
print("The Pearson Coefficient is ", pearson_coef, " with a p-value of P = ", p_value)
pearson_coef, p_value = stats.pearsonr(carDataframe['bore'],carDataframe['price'])
print("The Pearson Coefficient is ", pearson_coef, " with a p-value of P = ", p_value)
pearson_coef, p_value = stats.pearsonr(carDataframe['city-mpg'],carDataframe['price'])
print("The Pearson Coefficient is ", pearson_coef, " with a p-value of P = ", p_value)
pearson_coef, p_value = stats.pearsonr(carDataframe['highway-mpg'],carDataframe['price'])
print("The Pearson Coefficient is ", pearson_coef, " with a p-value of P = ", p_value)

# ANOVA Analysis of Variance
# Statistical method to test whether there are significant differences between the means of two or more groups.
# F-test. Calculates how much means deviate from the assumption that the means of all groups are the same. Larger score means larger difference.
# P-value. How statistically significant the calculated score is
# For example, if price variable is strongly correlated w variable we are analyzing, expect ANOVA to return sizeable F score and small p-value

groupedTest2 = df_gptest[['drive-wheels', 'price']].groupby(['drive-wheels'])
print(groupedTest2.get_group('4wd')['price'])
# examining all of the groups together.
f_val, p_val = stats.f_oneway(groupedTest2.get_group('fwd')['price'], groupedTest2.get_group('rwd')['price'],
                              groupedTest2.get_group('4wd')['price'])
print("ANOVA results: F=", f_val, ", P =", p_val)
# This is a great result, however, let's look at each group individually against price
f_val2, p_val2 = stats.f_oneway(groupedTest2.get_group('fwd')['price'], groupedTest2.get_group('rwd')['price'])
print("ANOVA results: F=", f_val2, ", P =", p_val2)
f_val3, p_val3 = stats.f_oneway(groupedTest2.get_group('4wd')['price'], groupedTest2.get_group('rwd')['price'])
print("ANOVA results: F=", f_val3, ", P =", p_val3)
f_val4, p_val4 = stats.f_oneway(groupedTest2.get_group('4wd')['price'], groupedTest2.get_group('rwd')['price'])
print("ANOVA results: F=", f_val4, ", P =", p_val4)
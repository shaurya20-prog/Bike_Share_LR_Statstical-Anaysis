# Bike_Share_LR_Statstical-Anaysis

A bike-sharing system is a service in which bikes are made available for shared use to individuals on a short term basis for a price or free. Many bike share systems allow people to borrow a bike from a "dock" which is usually computer-controlled wherein the user enters the payment information, and the system unlocks it. This bike can then be returned to another dock belonging to the same system.


A US bike-sharing provider BoomBikes has recently suffered considerable dips in their revenues due to the ongoing Corona pandemic. The company is finding it very difficult to sustain in the current market scenario. So, it has decided to come up with a mindful business plan to be able to accelerate its revenue as soon as the ongoing lockdown comes to an end, and the economy restores to a healthy state. 


In such an attempt, BoomBikes aspires to understand the demand for shared bikes among the people after this ongoing quarantine situation ends across the nation due to Covid-19. They have planned this to prepare themselves to cater to the people's needs once the situation gets better all around and stand out from other service providers and make huge profits.


They have contracted a consulting company to understand the factors on which the demand for these shared bikes depends. Specifically, they want to understand the factors affecting the demand for these shared bikes in the American market. The company wants to know:

Which variables are significant in predicting the demand for shared bikes.
How well those variables describe the bike demands
Based on various meteorological surveys and people's styles, the service provider firm has gathered a large dataset on daily bike demands across the American market based on some factors. 


Business Goal:
You are required to model the demand for shared bikes with the available independent variables. It will be used by the management to understand how exactly the demands vary with different features. They can accordingly manipulate the business strategy to meet the demand levels and meet the customer's expectations. Further, the model will be a good way for management to understand the demand dynamics of a new market. 

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


df=pd.read_csv('day.csv')

df

df['season'].unique()

df['yr'].unique()

df['weathersit'].unique()

df['holiday'].unique()

df['weekday'].unique()

sns.pairplot(df)

import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(20, 12))
plt.subplot(2,3,1)
sns.boxplot(x = 'weekday', y = 'cnt',  hue = 'holiday',data = df)
plt.subplot(2,3,2)
sns.boxplot(x = 'weathersit', y = 'cnt',hue = 'holiday', data = df)
plt.subplot(2,3,3)
sns.boxplot(x = 'season', y = 'cnt',hue = 'holiday', data = df)
plt.show()

#people in year 2019 on last day of the week(6th day) are taking more bikes as compared to count in 2018 .
#more People in 2019 are actually preferring more Clear, Few clouds, Partly cloudy, Partly cloudy weather as compared to 2018.
#more people in 2019 are actually preferring fall season  but also lot many people also prefer winter season as compared to count in 2018.
plt.figure(figsize=(20, 12))
plt.subplot(2,3,1)
sns.boxplot(x = 'weekday', y = 'cnt',  hue = 'yr',data = df)
plt.subplot(2,3,2)
sns.boxplot(x = 'weathersit', y = 'cnt',hue = 'yr', data = df)
plt.subplot(2,3,3)
sns.boxplot(x = 'season', y = 'cnt',hue = 'yr', data = df)
plt.show()

plt.figure(figsize = (16, 10))
sns.heatmap(df.corr(), annot = True, cmap="YlGnBu")
plt.show()
#temp and atemp are multicolinear, which can cause redundancy further.

df.drop("atemp", axis=1, inplace=True)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             

share=  pd.get_dummies(df['season'], prefix='season',drop_first = True)
share                                            

share_2=pd.get_dummies(df['weekday'],prefix='weekday_1',drop_first = True)
share_2

share_3=pd.get_dummies(df['weathersit'],prefix='weather_lol',drop_first = True)
share_3

total=pd.concat([df,share,share_2,share_3],axis=1)

total.columns

total.info()

total.drop(['weekday','season','weathersit','instant','registered','casual'], axis = 1, inplace = True)

total.info()

from sklearn.model_selection import train_test_split

# We specify this so that the train and test data set always have the same rows, respectively
np.random.seed(0)
df_train, df_test = train_test_split(total, train_size = 0.7, test_size = 0.3, random_state = 100)

df.info()

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

# Apply scaler() to all the columns except the 'yes-no' and 'dummy' variables
num_vars = ['temp', 'hum', 'windspeed','cnt','mnth']

df_train[num_vars] = scaler.fit_transform(df_train[num_vars])

df_train.head()

# Let's check the correlation coefficients to see which variables are highly correlated

plt.figure(figsize = (16, 10))
sns.heatmap(df_train.corr(), annot = True, cmap="YlGnBu")
plt.show()

had to drop registered and casual since according to the corr matrix, registered was the one which seemed a lot collinear.

As you might have noticed, temp seems to the correlated to cnt the most. Let's see a pairplot for cnt vs temp.

plt.figure(figsize=[6,6])
plt.scatter(df_train.temp, df_train.cnt)
plt.show()

# Dividing into X and Y sets for the model building

y_train = df_train.pop('cnt')
X_train = df_train

import statsmodels.api as sm

# Add a constant
X_train_lm = sm.add_constant(X_train[['temp']])

# Create a first fitted model
lr = sm.OLS(y_train, X_train_lm).fit()

# Check the parameters obtained

lr.params

# Let's visualise the data with a scatter plot and the fitted regression line
plt.scatter(X_train_lm.iloc[:, 1], y_train)
plt.plot(X_train_lm.iloc[:, 1],  0.169798 + 0.639952*X_train_lm.iloc[:, 1], 'r')
plt.show()

# Print a summary of the linear regression model obtained
print(lr.summary())

# Adding another variable

# Assign all the feature variables to X
X_train_lm = X_train[['temp', 'yr']]

# Build a linear model

import statsmodels.api as sm
X_train_lm = sm.add_constant(X_train_lm)

lr_1 = sm.OLS(y_train, X_train_lm).fit()

lr_1.params

# Check the summary
print(lr_1.summary())

As we can see the value of the Adj. R-squared has significantly gone up


# Assign all the feature variables to X
X_train_lm = X_train[['temp', 'yr','mnth']]


# Build a linear model

import statsmodels.api as sm
X_train_lm = sm.add_constant(X_train_lm)

lr_3 = sm.OLS(y_train, X_train_lm).fit()

lr_3.params

# Check the summary
print(lr_3.summary())

The adjusted r sqaure has improved from .689 to .714

X_train_lm=X_train[['temp','mnth','yr','holiday']]

# Build a linear model

import statsmodels.api as sm
X_train_lm = sm.add_constant(X_train_lm)

lr_4 = sm.OLS(y_train, X_train_lm).fit()

lr_4.params

lr_4.summary()

increase in Adjusted R squared can be seen

but the pvalue has been more than 0 for holiday

# 'weekday_1_weekday' 'weather_lol_weathersit'

# Giving all the columns

X_train.info()

X_train.drop(columns=['dteday'],inplace=True)

#Build a linear model

import statsmodels.api as sm
X_train_lm = sm.add_constant(X_train)

lr_5 = sm.OLS(y_train, X_train_lm).fit()

lr_5.params

X_train_lm

# X_train_lm.dropna(subset=['weekday_1_weekday','weather_lol_weathersit'], inplace=True)

lr_5.summary()

# Check for the VIF values of the feature variables. 
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.DataFrame()
vif['Features'] = X_train.columns
vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif

vif


X = X_train.drop('weekday_1_4',1,)

X.describe()

# Build a third fitted model
X_train_lm = sm.add_constant(X)

lr_6 = sm.OLS(y_train, X_train_lm).fit()

# Print the summary of the model
print(lr_6.summary())

# Calculate the VIFs again for the new model

vif = pd.DataFrame()
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif

mnth has high vif and p value

#low vif but high p value
X = X.drop('weekday_1_3', 1,)

# Build a third fitted model
X_train_lm = sm.add_constant(X)

lr_7 = sm.OLS(y_train, X_train_lm).fit()

lr_7.summary()

# Calculate the VIFs again for the new model

vif = pd.DataFrame()
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif

# low  vif value of 'weekday_1_5' but high p value

X=X.drop('weekday_1_5',1,)

# Build a third fitted model
X_train_lm = sm.add_constant(X)

lr_8 = sm.OLS(y_train, X_train_lm).fit()
lr_8.summary()

vif = pd.DataFrame()
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif

# 'mnth' being removed


# Build a third fitted model
X=X.drop('mnth',1,)
X_train_lm = sm.add_constant(X)

lr_9 = sm.OLS(y_train, X_train_lm).fit()
lr_9.summary()

vif = pd.DataFrame()
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif

# Build a third fitted model
X=X.drop('weekday_1_2',1,)
X_train_lm = sm.add_constant(X)

lr_10 = sm.OLS(y_train, X_train_lm).fit()
lr_10.summary()

vif = pd.DataFrame()
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif

X=X.drop('holiday',1,)
X_train_lm = sm.add_constant(X)

lr_11 = sm.OLS(y_train, X_train_lm).fit()
lr_11.summary()

vif = pd.DataFrame()
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif

X=X.drop('temp',1,)
X_train_lm = sm.add_constant(X)

lr_12 = sm.OLS(y_train, X_train_lm).fit()
lr_12.summary()

vif = pd.DataFrame()
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif

X=X.drop('hum',1,)
X_train_lm = sm.add_constant(X)

lr_13 = sm.OLS(y_train, X_train_lm).fit()
lr_13.summary()

vif = pd.DataFrame()
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif.Features

X=X.drop('weekday_1_1',1,)
X_train_lm = sm.add_constant(X)

lr_14 = sm.OLS(y_train, X_train_lm).fit()
lr_14.summary()

vif = pd.DataFrame()
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif.Features

y_train_price = lr_14.predict(X_train_lm)

# Plot the histogram of the error terms
fig = plt.figure()
sns.distplot((y_train - y_train_price), bins = 20)
fig.suptitle('Error Terms', fontsize = 20)                  # Plot heading 
plt.xlabel('Errors', fontsize = 18)       

num_vars = ['temp', 'hum', 'windspeed','cnt','mnth']

df_test[num_vars] = scaler.transform(df_test[num_vars])




df_test.info()

y_test = df_test.pop('cnt')
X_test = df_test

# Adding constant variable to test dataframe
X_test_m4 = sm.add_constant(X_test)

X_test_m4


X_test_m4 = X_test_m4.drop(["dteday","mnth","hum","weekday_1_1","weekday_1_2","weekday_1_3","weekday_1_4","weekday_1_5"], axis = 1)

X_test_m4 = X_test_m4.drop(['holiday'],axis=1)

X_test_m4 = X_test_m4.drop(['temp'],axis=1)

y_pred_m4 = lr_14.predict(X_test_m4)

# Plotting y_test and y_pred to understand the spread

fig = plt.figure()
plt.scatter(y_test, y_pred_m4)
fig.suptitle('y_test vs y_pred', fontsize = 20)              # Plot heading 
plt.xlabel('y_test', fontsize = 18)                          # X-label
plt.ylabel('y_pred', fontsize = 16)      

cnt=0.2471*yr+0.0566*workingday-0.1768*windspeed+0.2552*season_2+0.3138*season_3+0.2283*season_4+0.0640*season_4-0.0890*weather_lol_2-0.2960*weather_lol_3+0.2340


lr_14.summary()

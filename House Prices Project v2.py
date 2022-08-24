# Perhaps one of the most active areas of the economy at the moment is the price of houses. For the first time
# ever, the median average for rent in the United States surpassed $2,000. With the increasing demand for housing, the
# supply has simply been unable to keep up, due to rising labor costs, supply shortages, and inflation. As a result,
# housing prices have absolutely sky-rocketed. The purpose of this Kaggle competition project is to construct a machine
# learning algorithm that reliably prices the value of a home based upon the features given to us in our dataset with
# minimal inaccuracy.

# For the purposes of building out my project portfolio, we will utilize Linear Regression in order to showcase I can
# perform this quintessential modeling technique. We first begin by importing our packages:
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

# Again, the first thing we do is define our file paths to the data sets. Kaggle has split the datasets up for us into
# training and testing sets, however, we may split this up even further if need be using train-test-split.
housing_test = pd.read_csv(r'C:\Python\Data Sets\housing_test.csv', index_col='Id')
housing_train = pd.read_csv(r'C:\Python\Data Sets\housing_train.csv', index_col='Id')

# Because I do most of my programming inside the PyCharm IDE, these simply help with the visualizations of the data
# frames.
pd.set_option('display.max_columns', 80)  # This allows us to view all the column within our PyCharm IDE
pd.set_option('display.width', 1000)  # Formats our rows correctly.
pd.set_option('display.max_rows', 82)

# Any project should start with data exploration. Let's get an idea of what the shape of our data looks like to begin to
# understand which features may be the most helpful.
print(housing_train.head(3))
print("\n")
print(housing_train.shape)
print("\n")

# Notice that for the data set provided, a few columns could be considered ordinal variables. Recall that ordinal
# variables are ones that take on a specific "category." In the documentation for our data set, a lot of these variables
# are ranked on a scale. We will deal with them later, but for now, let's separate the numerical variables form the
# categorical ones.
data_types = {}
for column in housing_train.columns:
    data_types[column] = housing_train[column].dtype
print(data_types)  # Notice that columns that have can be considered ordinal variables have a datatype of 'O'
print("\n")

# We separate the numerical and categorical variables below.
categorical_vars_train = [column for column in housing_train.columns if housing_train[column].dtype == 'O']
numerical_vars_train = [column for column in housing_train.columns if housing_train[column].dtype != 'O']
print("There are", len(categorical_vars_train), "ordinal variables.")  # Check the length of the ordinal variables
print("There are", len(numerical_vars_train), "numerical variables.")  # Check the length of the numerical variables

# Next, we will proceed to the data cleaning process. Obviously, no real life data set will contain 0 NaN values.
# Let's first begin by checking the percentage of NaN values that are in each numerical column:


def null_counter(series):
    null_values = series.isnull().sum()  # Numerator is the number of null values in the argument series
    percentage = (null_values / len(series)) * 100  # Calculates the percentage of null values in a series
    return percentage


for column in numerical_vars_train:  # Iterate through our numerical variables
    nan_value = null_counter(housing_train[column])  # Apply our null counter function to each variable
    if nan_value > 0.0:
        print(f"The {column} column contains {nan_value} percent null values")  # Only return variables with > 0% nan's

# It appears we have 3 variables that contain NaN values: LotFrontage, MasVnrArea, and GarageYrBlt. In order to decide
# if we should use mean/median/mode imputation, we should find out which variables are discrete and which are continuous
continuous_variables = []
discrete_variables = []

for column in numerical_vars_train:  # We iterate through our numerical variables
    if len(housing_train[column].unique()) >= 20:  # We define a discrete variables as having less than 20 values
        continuous_variables.append(column)
    else:
        discrete_variables.append(column)  # Variables with more than 20 values will be considered discrete

print(f'There are {len(continuous_variables)} continuous variables')
print(f'There are {len(discrete_variables)} discrete variables')

print("\n")
# Now that we have defined our continuous and discrete variables, we can make a determination as to which imputation
# technique we wish to use. If it is discrete, we will use a median imputation. If continuous, we will use the mean.
for variable in ['LotFrontage', 'MasVnrArea', 'GarageYrBlt']:
    if variable in continuous_variables:
        print(f'{variable} is continuous')
    else:
        print(f'{variable} is discrete')

print("\n")
# All the variables are continuous. We shall use the mean to replace their NaN values
housing_train['LotFrontage'].replace(np.nan, housing_train['LotFrontage'].mean(), inplace=True)
housing_train['MasVnrArea'].replace(np.nan, housing_train['MasVnrArea'].mean(), inplace=True)
housing_train['GarageYrBlt'].replace(np.nan, housing_train['GarageYrBlt'].mean(), inplace=True)

print("\n")
# Now we proceed to our categorical/ordinal variables. We will begin in a similar fashion as the numerical variables in
# that we will see how many NaN values we currently have to deal with.
for column in categorical_vars_train:
    nulls_value = null_counter(housing_train[column])
    if nulls_value > 0.0:
        print(f'The {column} column contains {nulls_value} percent null values')

very_bad = [column for column in categorical_vars_train if null_counter(housing_train[column]) > 80]

print("\n")
# Our very bad list has columns that contain more than 80% of null values. There are 4 of them: Alley, PoolQC, Fence,
# and MiscFeature. It will be up to us if we want to decide to use these or drop them when we create our model.

# Now comes the most challenging part: how do we quantify the values within our categorical variables? We cannot simply
# plug in a non-numeric value into our regression model. One way to approach this is to map a numeric value to each
# categorical value. However, say if we assigned a value of 5 to basement in "Excellent" condition and a 1 to a variable
# in "Poor" condition, how do we know that the Excellent basement is 5x better than the Poor one? Couldn't it be only
# 2x better, or perhaps 50x better?

# As a result, we will create a mapping for the categorical variables, however, we shall create a function below that
# essentially creates a relationship for categorical variables based upon the sale price. Whichever variable value has
# a higher mean price, to that we shall assign a number between 0 and n, where n is the number of possible values. This
# allows us to introduce less bias by randomly picking numbers between 0 and n to assign to each variable value.


def train_categorical_encoding(var, target):
    ordered_labels = housing_train.groupby([var])[target].mean().sort_values().index
    categorical_mapping = {k: i for i, k in enumerate(ordered_labels, 0)}

    housing_train[var] = housing_train[var].map(categorical_mapping)
    housing_test[var] = housing_test[var].map(categorical_mapping)


# Before we apply our function to our categorical variables, we will replace the Null values with the word "Missing".
# This way, we won't have to worry about our loop breaking down.
for column in categorical_vars_train:
    housing_train[column].replace(np.nan, 'Missing', inplace=True)

for column in categorical_vars_train:
    housing_test[column].replace(np.nan, 'Missing', inplace=True)

# We now apply our function to each categorical column.
for column in categorical_vars_train:
    train_categorical_encoding(column, 'SalePrice')

# We now check to see if our transformation worked (it does)
print(housing_train.head())

print("\n")
# Let's check the status of our null values for our data frame. (There should be none)
print(housing_train.isnull().sum())

# We now will plot a correlation matrix. I have tried repeatedly to incorporate every variable to no avail. This makes
# sense from a theoretical perspective. Incorporating each variable into our feature matrix introduces a tremendous
# amount of bias into our algorithm. We need to incorporate the strongest variables in order to correct our variance.
# As a result, we shall incorporate variables that have a correlation coefficient greater than .3 into our training
# matrix.
correlation_series = housing_train.corr()['SalePrice'].sort_values(ascending=False)  # Each value is the r-value
print(correlation_series)
variables_of_interest = ['OverallQual', 'Neighborhood', 'GrLivArea', 'ExterQual', 'KitchenQual', 'GarageCars',
                         'BsmtQual', 'GarageArea', 'TotalBsmtSF', '1stFlrSF', 'FullBath', 'GarageFinish', 'TotRmsAbvGrd',
                         'FireplaceQu', 'YearBuilt', 'YearRemodAdd', 'Foundation', 'GarageType', 'GarageYrBlt',
                         'Fireplaces', 'OpenPorchSF', 'MasVnrType', 'HeatingQC', 'MasVnrArea', 'LotArea', 'BsmtFinType1',
                         'BsmtExposure', 'Exterior1st', 'SaleType', 'Exterior2nd', 'LotFrontage', 'MSZoning',
                         'WoodDeckSF', '2ndFlrSF', 'OpenPorchSF']

print("\n")
# We will now define our feature matrix and target variable. From there we will construct our model using linear
# regression and run a quick mean squared error on the training set to ensure our model fits appropriately.
X_train = housing_train[variables_of_interest]  # Defining our training matrix
Y_train = housing_train['SalePrice']  # Defining our training target variable

# Before we scale our data, we should consider the outliers within the columns we have chosen for our training set. We
# can create a boxplot for each column and note which columns contain a lot of outliers.

# After plotting box-plots, we notice that the following columns have a fair amount of outliers:
# GrLivArea, GarageArea, TotalBsmtSF, 1stFlrSF
# To deal with these, we will perform a process whereby we will replace these outlier values with the mean of these
# respective columns.
outlier_train = ['GrLivArea', 'GarageArea', 'TotalBsmtSF', '1stFlrSF']
for column in outlier_train:
    Q1_train = X_train[column].quantile(.25)
    Q3_train = X_train[column].quantile(.75)
    IQR_train = Q3_train - Q1_train
    high = Q3_train + (1.5*IQR_train)
    low = Q1_train - (1.5*IQR_train)
    X_train.loc[X_train[column] > high, column] = X_train[column].mean()
    X_train.loc[X_train[column] < low, column] = X_train[column].mean()

# We use standard scaler for our model to make our variables more normally distributed
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)

lin_reg = LinearRegression()
lin_reg.fit(X_train_s, Y_train)  # Creating our model
train_predictions = lin_reg.predict(X_train)  # Applying our model to the training data to make sure it is accurate


# ================================================ END OF TRAINING =================================================== #

# ============================================ BEGIN TEST PREPROCESSING ============================================== #
# Now it is time to apply the exact same methods above to our test data. Recall, the way Kaggle conducts competitions
# is by issuing 2 csv data files for training and testing. Once the model has been created, we must apply it to the
# training set and submit it to Kaggle to receive our final score. As a result, I won't go into too much detail in the
# comments, and only comment where more clarification is needed.

# We will use the median to replace our discrete variables and the mean for our continuous variables.
dirty_continuous = ['LotFrontage', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'GarageYrBlt',
                    'GarageArea']
dirty_discrete = ['BsmtFullBath', 'BsmtHalfBath', 'GarageCars']

# We then will loop through each dirty column and impute the mean or median.
for column in dirty_continuous:
    housing_test[column].replace(np.nan, housing_test[column].mean(), inplace=True)

for column in dirty_discrete:
    housing_test[column].replace(np.nan, housing_test[column].median(), inplace=True)


# Now, let us check if we still have any null values (we do)
print(housing_test.isnull().sum())

# It's safe to assume that these remaining null values are remnants from the categorical mapping process. As such, we
# will impute the median for these values.
leftovers = ['MSZoning', 'Utilities', 'Exterior1st', 'Exterior2nd', 'KitchenQual', 'Functional', 'SaleType']
for column in leftovers:
    housing_test[column].replace(np.nan, housing_test[column].median(), inplace=True)

# Check one last time (all good)
print('\n')
print(housing_test.isnull().sum())
print(housing_test.head())

# Similarly, we will remove outliers from our test dataset to (hopefully) improve our accuracy.

# We have completely cleaned our test data set, normalized the continuous variables, and performed categorical to
# numeric mapping. We are ready to use this data set to make our predictions.
X_test = housing_test[variables_of_interest]

# We now deal with the outliers in our test set the same way we dealt with them in our training set.
for column in outlier_train:
    Q1_test = X_test[column].quantile(.25)
    Q3_test = X_test[column].quantile(.75)
    IQR_test = Q3_test - Q1_test
    high = Q3_test + (1.5*IQR_test)
    low = Q1_test - (1.5*IQR_test)
    X_test.loc[X_test[column] > high, column] = X_test[column].mean()
    X_test.loc[X_test[column] < low, column] = X_test[column].mean()

# Again, we make our model more normally distributed.
X_test_s = scaler.fit_transform(X_test)
predictions = lin_reg.predict(X_test_s)

# We then write our responses to an Excel csv file and submit them to Kaggle to examine the accuracy rating
prediction_series = pd.Series(data=predictions, index=housing_test.index)

# We then write our submission to a CSV file for upload to Kaggle ot get our RMSE score
prediction_series.to_csv(path_or_buf=r'C:\Python\Data Sets\housing_prediction.csv', header=['SalePrice'])

# ============================================= CONCLUSION =========================================================== #
# After many iterations, the RMSE score for this project, according to the Kaggle competition, is .18442. This is a
# fairly accurate score, and certainly a score that I am comfortable placing on my project portfolio website. I've
# learned a lot over the course of this project. From learning how to handle categorical variables to understanding the
# sensitivity of regression models to outliers, I truly feel like the project propelled me forward in my career as a
# Machine Learning Engineer.

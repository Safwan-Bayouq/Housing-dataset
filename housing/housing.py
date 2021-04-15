#Needed libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

#This library is for removing the warnings
import warnings
warnings.filterwarnings("ignore")

#train and test datasets
train_data = pd.read_csv(r'~\datasets\housing\train.csv')
test_data = pd.read_csv(r'~\datasets\housing/test.csv')

#Droping the columns that have more than 50% of its values as NaN from the training set 
train_data = train_data.drop('Alley', axis=1)
train_data = train_data.drop('FireplaceQu', axis=1)
train_data = train_data.drop('PoolQC', axis=1)
train_data = train_data.drop('Fence', axis=1)
train_data = train_data.drop('MiscFeature', axis=1)

#Droping the columns that have more than 50% of its values as NaN from the testing set 
test_data = test_data.drop('Alley', axis=1)
test_data = test_data.drop('FireplaceQu', axis=1)
test_data = test_data.drop('PoolQC', axis=1)
test_data = test_data.drop('Fence', axis=1)
test_data = test_data.drop('MiscFeature', axis=1)

#Working with the numeric columns that have NaN values in the training dataset
imputer = SimpleImputer(strategy='mean')
train_data[['LotFrontage']] = imputer.fit_transform(train_data[['LotFrontage']])
train_data[['GarageYrBlt']] = imputer.fit_transform(train_data[['GarageYrBlt']])
train_data[['MasVnrArea']] = imputer.fit_transform(train_data[['MasVnrArea']])

#Working with the numeric columns that have NaN values in the testing dataset
test_data[['LotFrontage']] = imputer.fit_transform(test_data[['LotFrontage']])
test_data[['GarageYrBlt']] = imputer.fit_transform(test_data[['GarageYrBlt']])
test_data[['MasVnrArea']] = imputer.fit_transform(test_data[['MasVnrArea']])
test_data[['BsmtFinSF1']] = imputer.fit_transform(test_data[['BsmtFinSF1']])
test_data[['BsmtFinSF2']] = imputer.fit_transform(test_data[['BsmtFinSF2']])
test_data[['BsmtUnfSF']] = imputer.fit_transform(test_data[['BsmtUnfSF']])
test_data[['TotalBsmtSF']] = imputer.fit_transform(test_data[['TotalBsmtSF']])
test_data[['BsmtFullBath']] = imputer.fit_transform(test_data[['BsmtFullBath']])
test_data[['BsmtHalfBath']] = imputer.fit_transform(test_data[['BsmtHalfBath']])
test_data[['GarageCars']] = imputer.fit_transform(test_data[['GarageCars']])
test_data[['GarageArea']] = imputer.fit_transform(test_data[['GarageArea']])

#Working with the categorical columns that have NaN values in the training dataset
frequent_imputer = SimpleImputer(strategy='most_frequent')
train_data[['MasVnrType']] = frequent_imputer.fit_transform(train_data[['MasVnrType']])
train_data[['BsmtQual']] = frequent_imputer.fit_transform(train_data[['BsmtQual']])
train_data[['BsmtCond']] = frequent_imputer.fit_transform(train_data[['BsmtCond']])
train_data[['BsmtFinType1']] = frequent_imputer.fit_transform(train_data[['BsmtFinType1']])
train_data[['BsmtFinType2']] = frequent_imputer.fit_transform(train_data[['BsmtFinType2']])
train_data[['Electrical']] = frequent_imputer.fit_transform(train_data[['Electrical']])
train_data[['GarageType']] = frequent_imputer.fit_transform(train_data[['GarageType']])
train_data[['GarageFinish']] = frequent_imputer.fit_transform(train_data[['GarageFinish']])
train_data[['GarageQual']] = frequent_imputer.fit_transform(train_data[['GarageQual']])
train_data[['GarageCond']] = frequent_imputer.fit_transform(train_data[['GarageCond']])
train_data[['BsmtExposure']] = frequent_imputer.fit_transform(train_data[['BsmtExposure']])

#Working with the categorical columns that have NaN values in the testing dataset
test_data[['MasVnrType']] = frequent_imputer.fit_transform(test_data[['MasVnrType']])
test_data[['BsmtQual']] = frequent_imputer.fit_transform(test_data[['BsmtQual']])
test_data[['BsmtCond']] = frequent_imputer.fit_transform(test_data[['BsmtCond']])
test_data[['BsmtFinType1']] = frequent_imputer.fit_transform(test_data[['BsmtFinType1']])
test_data[['BsmtFinType2']] = frequent_imputer.fit_transform(test_data[['BsmtFinType2']])
test_data[['SaleType']] = frequent_imputer.fit_transform(test_data[['SaleType']])
test_data[['GarageType']] = frequent_imputer.fit_transform(test_data[['GarageType']])
test_data[['GarageFinish']] = frequent_imputer.fit_transform(test_data[['GarageFinish']])
test_data[['GarageQual']] = frequent_imputer.fit_transform(test_data[['GarageQual']])
test_data[['GarageCond']] = frequent_imputer.fit_transform(test_data[['GarageCond']])
test_data[['BsmtExposure']] = frequent_imputer.fit_transform(test_data[['BsmtExposure']])
test_data[['MSZoning']] = frequent_imputer.fit_transform(test_data[['MSZoning']])
test_data[['Utilities']] = frequent_imputer.fit_transform(test_data[['Utilities']])
test_data[['Exterior1st']] = frequent_imputer.fit_transform(test_data[['Exterior1st']])
test_data[['Exterior2nd']] = frequent_imputer.fit_transform(test_data[['Exterior2nd']])
test_data[['KitchenQual']] = frequent_imputer.fit_transform(test_data[['KitchenQual']])
test_data[['Functional']] = frequent_imputer.fit_transform(test_data[['Functional']])

#This function return the numeric columns from a dataset for scaling
def get_numeric_and_categorical_columns(data):
    numeric_columns = []
    categorical_columns = []
    
    #Getting all the numeric columns
    numeric = data.dtypes[data.dtypes != np.object]
    numeric = list(numeric.index)

    #Removing the binary columns
    for x in numeric:
        if float(np.max(data[x])) > 1:
            numeric_columns.append(x)
        
    #Getting all the categorical columns
    categorical_columns = data.dtypes[data.dtypes == np.object]
    categorical_columns = list(categorical_columns.index)

    return numeric_columns, categorical_columns

numeric_train, categorical_columns_train = get_numeric_and_categorical_columns(train_data)
numeric_test, categorical_columns_test = get_numeric_and_categorical_columns(test_data)

#Removing the SalePrice, because we do not want to scale the prediction columns
numeric_train.remove('SalePrice')
numeric_train.remove('Id')
numeric_test.remove('Id')

#Scaling the numeric columns using the MinMaxScaler
scaler = MinMaxScaler()

for x in numeric_train:
    train_data[x] = scaler.fit_transform(train_data[[x]])
    
for x in numeric_test:
    test_data[x] = scaler.fit_transform(test_data[[x]])
    
#Using OrdinalEncoder for the categorical columns
ordinal_encoder = OrdinalEncoder()

for x in categorical_columns_train:
    encoded_train = ordinal_encoder.fit_transform(train_data[[x]])
    train_data[x] = encoded_train
    
for x in categorical_columns_test:
    encoded_test = ordinal_encoder.fit_transform(test_data[[x]])
    test_data[x] = encoded_test
    
#Splitting the training dataset
y=train_data.iloc[:,[-1]]
x=train_data.drop(y.columns,axis = 1)

X_train, X_valid, y_train, y_valid = train_test_split(x, y, train_size=0.7, test_size=0.3,random_state=42)

def all_(X_train, X_valid, y_train, y_valid):
    
    #Creating a Learning Curve
    train_dct, train_rfr, train_xgb = list(), list(), list()
    
    values = [i for i in range(1, 15)]

    # evaluate a decision tree for each depth
    for i in values:
        # configure the model
        model = DecisionTreeRegressor(max_depth=i)

        # fit model on the training dataset
        model.fit(X_train, y_train)

        # evaluate on the validation dataset
        train_acc = model.score(X_valid, y_valid)
        train_dct.append(train_acc)

    # evaluate a RandomForestRegressor for each depth
    for i in values:
        # configure the model
        model = RandomForestRegressor(max_depth=i)

        # fit model on the training dataset
        model.fit(X_train, y_train)

        # evaluate on the validation dataset
        train_acc = model.score(X_valid, y_valid)
        train_rfr.append(train_acc)

    # evaluate XGBRegressor for each depth
    for i in values:
        # configure the model
        model = XGBRegressor(max_depth=i)

        # fit model on the training dataset
        model.fit(X_train, y_train)

        # evaluate on the validation dataset
        train_acc = model.score(X_valid, y_valid)
        train_xgb.append(train_acc)

        
    plt.plot(values, train_dct, '-o', label='DecisionTreeRegressor')
    plt.plot(values, train_rfr, '-o', label='RandomForestRegressor')
    plt.plot(values, train_xgb, '-o', label='XGBRegressor')
    plt.legend()
    plt.xlabel('C/max depth')
    plt.ylabel('accuracy')
    plt.show()
    
#Testing the Decision Tree algorithms in the validation data
dct = DecisionTreeRegressor()
dct.fit(X_train, y_train)
print('\nThe accuracy from the DecisionTreeRegressor algorithm =', dct.score(X_valid, y_valid))

#Testing the Linear Regression algorithms in the validation data
lr = LinearRegression()
lr.fit(X_train, y_train)
print('\nThe accuracy from the LinearRegression algorithm =', lr.score(X_valid, y_valid))

#Testing the Random Forest algorithms in the validation data
rfr = RandomForestRegressor()
rfr.fit(X_train, y_train)
print('\nThe accuracy from the RandomForestRegressor algorithm =', rfr.score(X_valid, y_valid))

#Testing the Extreme Gradient Boosting algorithms in the validation data
xgb = XGBRegressor()
xgb.fit(X_train, y_train)
print('\nThe accuracy from the XGBRegressor algorithm =', xgb.score(X_valid, y_valid))

all_(X_train, X_valid, y_train, y_valid)

#Testing the model on the testing dataset
xgb = XGBRegressor(max_depth=5)
xgb.fit(x, y)

predictions = xgb.predict(test_data)

print('\nThe predictions from the test set are:', predictions)

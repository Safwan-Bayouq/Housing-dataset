This code is for the housing dataset from the (Hands-On Machine Learning with Scikit-Learn & TensorFlow) book

-------------------

1) we dropped from the training and testing sets the columns that had more than 50% of its values as NaNs.

2) for the columns that have NaN values less than 50% of its values in the training and testing sets, we used
(SimpleImputer) to change the NaN values, as of the numeric columns, the strategy was 'mean', and for the categorical
columns, we used the 'most_frequent' strategy.

3) we get the names of the numeric and categorical columns from the training set and testing set for scaling and 
categorical encoding procedures.

4) we used (MinMaxScaler) for scaling the numeric columns and (OrdinalEncoder) for encoding the categorical variables.

5) we split the training data into 70% training and 30% validation.

6) we made a function that plots the tuning of the algorithms we used.

7) we used DecisionTreeRegressor, RandomForestRegressor, LinearRegression, SVR, and XGBRegressor

8) lastly, we predicted the SalePrice columns from the test set using XGBRegressor with max_depth set to 5.
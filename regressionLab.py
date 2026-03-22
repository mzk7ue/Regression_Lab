# %%
# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

# %%
# Question 1
# Load cleaned q1_clean data 
df = pd.read_csv('https://raw.githubusercontent.com/DS3001/linearRegression/refs/heads/main/data/Q1_clean.csv')

# Strip any whitespace from column names
df.columns = df.columns.str.strip()

# Inspect dataset structure
df.info()

# %%
# Question 1.1
# Compute the average prices and review scores by Neighbourhood
df_grouped = df.groupby('Neighbourhood')[['Price', 'Review Scores Rating']].mean()
df_grouped.head()

# %%
# Most expensive borough on average: calculated by sorting the grouped dataframe by Price in descending order
most_expensive = df_grouped.sort_values(by = 'Price', ascending = False).index[0]

# The most expensive borough on average is Manhattan.
print(most_expensive)

# %%
# Kernel density plot of price, grouped by neighborhood
sns.kdeplot(data = df, x = 'Price', hue = 'Neighbourhood')
plt.title('Kernel Density Plot of Price by Neighborhood')

# %%
# Kernel density plot of log price, grouped by neighborhood
df['log_price'] = np.log(df['Price'] + 1)
sns.kdeplot(data = df, x = 'log_price', hue = 'Neighbourhood')
plt.title('Kernel Density Plot of Log Price by Neighbourhood')

# %%
# Question 1.2
# Create dummy variables for all the neighborhoods
# In regression without an intercept, you should not drop any dummy variables (drop_first = False)
df_without = pd.get_dummies(df, columns = ['Neighbourhood'], drop_first = False, prefix = ['nbh'])
df_without.info()

# %%
# Define X (dummies), which are our predictors, and y (Price), our response variable
X_simple = df_without[['nbh_Bronx', 'nbh_Brooklyn', 'nbh_Manhattan', 'nbh_Queens', 'nbh_Staten Island']]
y_target = df_without['Price']

# Fitting a linear regession model without an intercept (fit_intercept = False)
model_without = LinearRegression(fit_intercept = False).fit(X_simple, y_target)

# %%
print(f"Without Intercept: Coefficient = {model_without.coef_}, Intercept = {model_without.intercept_:.2f}, R² = {model_without.score(X_simple, y_target):.4f}")

# %%
# Interpretation: 
# Pattern: The average price of each of the neighborhood is the same as its corresponding coefficient.
 
# We can conclude that the coefficients in a regression of a continuous variable on one categorical 
# variable are the same as the average of the variable for each of the levels in that categorical 
# variable (in this case, the average price).

# %% 
# Question 1.3
# Create dummy variables, dropping one category (Bronx) using drop_first = True
df_with = pd.get_dummies(df, columns = ['Neighbourhood'], drop_first = True, prefix = ['nbh'])
df_with.info()

# By leaving an intercept in the linear model, we have to set drop_first = True instead of False when creating 
# our dummies (dropping the base level - Bronx) to avoid the dummy variable trap. 
# Having an intercept will account for the base level; the intercept we get is the average price of Bronx, 
# and it is the first coefficient value that we got previously in the regression model without intercept.

# %%
# Define X and y
X_simple_with = df_with[['nbh_Brooklyn', 'nbh_Manhattan', 'nbh_Queens', 'nbh_Staten Island']]
y_target_with = df_with['Price']

# %%
# Fitting a linear regession model with an intercept (fit_intercept = True)
model_with = LinearRegression(fit_intercept=True).fit(X_simple_with, y_target_with)
print(f"With Intercept: Coefficient = {model_with.coef_}, Intercept = {model_with.intercept_:.2f}, R² = {model_with.score(X_simple_with, y_target_with):.4f}")

# The intercept is the average price of Bronx (75.28). 
# Given that the neighborhood is Brooklyn, the average price is 52.47 higher than Bronx.
# Given that the neighborhood is Manhattan, the average price is 108.39 higher than Bronx.
# Given that the neighborhood is Queens, the average price is 21.58 higher than Bronx.
# Given that the neighborhood is Staten Island, the average price is 70.89 higher than Bronx.

# To get the coefficients in question 1.2 (part 2) from these new coefficients, you add the intercept to each of the new coefficients.
# For Bronx, the intercept is the coefficient from part 2.

# %%
# Question 1.4
# Similar to the previous X, but includes Review Scores Rating
X = df_with[['Review Scores Rating', 'nbh_Brooklyn', 'nbh_Manhattan', 'nbh_Queens', 'nbh_Staten Island']]
y = df_with['Price']

# Splitting 80/20 into the train and test set 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# Fit the model
split_model = LinearRegression().fit(X_train, y_train)

# Predictions
y_pred = split_model.predict(X_test)

# Mean Squared Error
mse = mean_squared_error(y_test, y_pred)

# Root Mean Squared Error: average prediction error
rmse = np.sqrt(mse)
print(f"Root Mean Squared Error: {rmse:.2f}")

# R-Squared: proportion of varaince in price that is explained by the model
r_squared = r2_score(y_test, y_pred)
print(f"R² Score: {r_squared:.4f}")

# %%
print(f"Coefficient on Review Scores Rating = {split_model.coef_[0]:.4f}")

# %%
print(f"Coefficients: {split_model.coef_}")
# The most expensive kind of property you can rent is in Manhattan.
# Since the intercept that is added to the coefficient is the same for all the neighborhoods, 
# Manhattan will have the largest value and therefore be the most expensive kind of/location for property.

# %%
# Question 1.5
# One-hot encoding of the Property Type variable
df_prop = pd.get_dummies(df_with, columns = ['Property Type'], drop_first = True, prefix = ['Ty'])

# %%
# Look at column names 
df_prop.head()
df_prop.columns

# %%
# Define X: Review Scores Rating + all the neighborhoods + all the property types
X_prop = df_prop[['Review Scores Rating', 'nbh_Brooklyn', 'nbh_Manhattan', 'nbh_Queens', 'nbh_Staten Island', 
            'Ty_Bed & Breakfast', 'Ty_Boat', 'Ty_Bungalow', 'Ty_Cabin', 
            'Ty_Camper/RV', 'Ty_Castle', 'Ty_Chalet', 'Ty_Condominium', 'Ty_Dorm', 
            'Ty_House', 'Ty_Hut', 'Ty_Lighthouse', 'Ty_Loft', 'Ty_Other',
            'Ty_Townhouse', 'Ty_Treehouse', 'Ty_Villa']]

y_prop = df_prop['Price']

X_prop_train, X_prop_test, yp_train, yp_test = train_test_split(X_prop, y_prop, test_size = 0.2, random_state = 42)

model_prop = LinearRegression().fit(X_prop_train, yp_train)

y_pred_prop = model_prop.predict(X_prop_test)

# Mean Squared Error
mse_prop = mean_squared_error(yp_test, y_pred_prop)

# Root Mean Squared Error
rmse_prop = np.sqrt(mse_prop)
print(f"Root Mean Squared Error: {rmse_prop:.2f}")

# R-Squared
r_squared_prop = r2_score(yp_test, y_pred_prop)
print(f"R² Score: {r_squared_prop:.4f}")

# %%
print(f"Coefficient on Review Scores Rating = {model_prop.coef_[0]:.4f}")

# %%
print(f"Coefficients: {model_prop.coef_}")
# The most expensive kind of property you can rent is Bungalow.
# We can determine this by comparing all the coefficients printed above, 
# and finding the highest predicted prices for each property type, 
# which is found by adding the intercept to the corresponding coefficient.

# %%
# Question 1.6 
# Coefficient on Review Scores Rating for Part 4: 1.2119
# Coefficient on Review Scores Rating for Part 5: 1.2010

# For Part 4, the coefficient on Review Scores Rating is saying that 
# the change in price for a unit increase in Review Scores Rating, 
# holding Neighborhood constant, while Part 5 is holding Neighborhood AND Property Type constant.
# There is a small change in the coefficient on Review Scores Rating between Part 4 and 5, 
# which means that Property Type has little to no impact on Review Scores Rating.

# %% 
# Question 2
# Read in the cars dataset
cars = pd.read_csv('cars_hw.csv')
cars.info()

# %%
# Question 2.1
# Remove the index column (unnecessary)
cars = cars.iloc[:, 1:]
cars.info()

# %%
# Check distribution of our chosen Y variable 
plt.hist(cars['Price'])

# %%
# Since Price is right skewed, we should transform
cars['log_Price'] = np.log(cars['Price'] + 1)
plt.hist(cars['log_Price'])

# %%
# Remove the outliers for log-Price using IQR
Q1 = cars['log_Price'].quantile(0.25)
Q3 = cars['log_Price'].quantile(0.75)
IQR = Q3 - Q1

cars = cars[(cars['log_Price'] >= Q1 - 1.5*IQR) & (cars['log_Price'] <= Q3 + 1.5*IQR)]

# It dropped two rows 
cars.info()

# %%
# Question 2.2
# Grouping the cars by brand, and looking at price
summary = cars.groupby('Make')['Price'].describe()
summary

# %%
# Kernel Density Plot of cars by the brand
sns.kdeplot(data = cars, x = 'Price', hue = 'Make')
plt.title("kernel Density Plot of Car Price by Make")

# The top most expensive brands are MG Motors, Kia, and Jeep. 
# These three brands are the only car models that have a minimum of 
# a million dollars (based on min) and have the highest mean. 

# %%
# Question 2.3
X_cars = cars.drop(columns = ['Price', 'log_Price'])
y_cars = cars['log_Price']

X_cars_train, X_cars_test, yc_train, yc_test = train_test_split(X_cars, y_cars, test_size = 0.2, random_state = 42)

print(X_cars_train.shape, X_cars_test.shape)
print(yc_train.shape, yc_test.shape)

# %%
# Question 2.4

# Numeric Variables Only Regression Model
numeric = cars[['Mileage_Run', 'Seating_Capacity', 'Make_Year']]

# Fit a linear regression model with only numeric features
Xn_train, Xn_test, yn_train, yn_test = train_test_split(numeric, y_cars, test_size = 0.2, random_state = 42)

model_num = LinearRegression().fit(Xn_train, yn_train)

yn_pred = model_num.predict(Xn_test)

# Mean Squared Error
mse_n = mean_squared_error(yn_test, yn_pred)

# Root Mean Squared Error
rmse_n = np.sqrt(mse_n)
print(f"Root Mean Squared Error: {rmse_n:.2f}")

# R-Squared
r_squared_n = r2_score(yn_test, yn_pred)
print(f"R² Score: {r_squared_n:.4f}")

# %%
# Categorical Variables Only Regression Model
cat_cols = cars.select_dtypes(include = ['object', 'string']).columns.tolist()
cat_cols

# %%
# One-hot encode, with drop_first = True to avoid dummy trap
encoded = pd.get_dummies(cars[cat_cols], drop_first = True, prefix = ['C', 'BT', 'MR', 'Num', 'FT', 'T', 'TT'])

# Fit a linear regression model with only categorical features
X_cat_train, X_cat_test, y_cat_train, y_cat_test = train_test_split(encoded, y_cars, test_size = 0.2, random_state = 42)

model_cat = LinearRegression().fit(X_cat_train, y_cat_train)

y_cat_pred = model_cat.predict(X_cat_test)

# Mean Squared Error
mse_cat = mean_squared_error(y_cat_test, y_cat_pred)

# Root Mean Squared Error
rmse_cat = np.sqrt(mse_cat)
print(f"Root Mean Squared Error: {rmse_cat:.2f}")

# R-Squared
r_squared_cat = r2_score(y_cat_test, y_cat_pred)
print(f"R² Score: {r_squared_cat:.4f}")

# The model with only categorical variables performed better in 
# the test set because it has a lower RMSE and a higher R-squared value. 
# There is a higher percent of variability explained by the categorical-only model.

# %%
# All predictors Regression Model
all = pd.concat([numeric, encoded], axis = 1)

X_all_train, X_all_test, y_all_train, y_all_test = train_test_split(all, y_cars, test_size = 0.2, random_state = 42)

model_all = LinearRegression().fit(X_all_train, y_all_train)

y_all_pred = model_all.predict(X_all_test)

# Mean Squared Error
mse_all = mean_squared_error(y_all_test, y_all_pred)

# Root Mean Squared Error
rmse_all = np.sqrt(mse_all)
print(f"Root Mean Squared Error: {rmse_all:.2f}")

# R-Squared
r_squared_all = r2_score(y_all_test, y_all_pred)
print(f"R² Score: {r_squared_all:.4f}")

# The joint model performs better than the individual 
# ones by ~15% more variability explained compared to the 
# categorical only (the higher of the two individual models).

# %%
# Question 2.5
# Fit a regression model using numeric variables
X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(numeric, y_cars, test_size = 0.2, random_state = 42)

results = {}

for degree in range(1,11):
    pf = PolynomialFeatures(degree = degree, include_bias = False)
    Xtr = pf.fit_transform(X_train_p)
    Xte = pf.transform(X_test_p)
    m = LinearRegression().fit(Xtr, y_train_p)
    y_pred = m.predict(Xte)
    rmse = np.sqrt(mean_squared_error(y_test_p, y_pred))
    r2  = m.score(Xte, y_test_p)
    results[f'degree_{degree}'] = r2
    print(f"Degree {degree}  |  Test R²: {r2:.4f} | Test RMSE: {rmse:.4f}")

# As the degree increases, R-squared increases then decreases, and RMSE increases. 
# R-squared decreases/goes negative at degree 4 on the test set.
# Our best model with expanded features would be at degree 3 with an R-squared of 0.4305 and RMSE of 0.3277.
# This is significantly worse than our best model from Part 4 (the joint model) because the joint model had a 
# higher R-squared and lower RMSE.

# %%
# Question 2.6

# The best model was the joint model.
plt.figure(figsize=(6, 6))
plt.scatter(y_all_test, y_all_pred, alpha=0.5)
plt.plot([y_all_test.min(), y_all_test.max()], [y_all_test.min(), y_all_test.max()], 'r--', lw=2)
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('True vs. Predicted Values')
plt.show()

# The predicted values and true values roughly line up along the diagonal. 

# %%
# Residual Analysis
residuals = y_all_test - y_all_pred

# kernel Density Plot of the residuals
sns.kdeplot(residuals)
plt.title("Kernel Density Plot of Residuals")
plt.xlabel('Residuals')
plt.show()

# Yes, the residuals look roughly bell-shaped around 0.

# %%
# Strengths:
# Our R-squared value is relatively high with roughly 77% of variability in the model explained.
# The kernel density plot of residuals also shows a roughly bell-shaped distribution around 0, which means that the model predictions are unbiased on average.
# Our RMSE is also on the lower end at 0.21.

# Weaknesses:
# We have several one-hot encoded features, which increased the number of predictors significantly. This means that the model might not generalize as well, and that our model could be overfitted.
# While the residuals are roughly bell-shaped, there may still be outliers or extreme residuals, which is indicated in the tail in the model. This means that the model may struggle with predicting very expensive or very cheap cars.

# %%
# Question 3

# %% 
# Question 3.1
# Source: https://www.kaggle.com/datasets/budincsevity/szeged-weather?resource=download
weather = pd.read_csv('weatherHistory.csv')
weather.info()

# %%
# Question 3.2
# Check for duplicates + drop
weather.duplicated().sum()
weather = weather.drop_duplicates()
weather.info()

# %%
# Check for missing values + drop
weather.isnull().sum()
weather = weather.dropna()
weather.info()

# %%
# Remove the date column
weather = weather.iloc[:, 1:]
weather.info()

# %%
# Check distribution of Y variable 
plt.hist(weather['Apparent Temperature (C)'])

# %%
# EDA for Precipitation Type, Humidity, and Visibility 
plt.hist(weather['Precip Type'])

# %%
plt.hist(weather['Humidity'])

# %%
plt.hist(weather['Visibility (km)'])

# %%
# Identifying the predictors for each model
# Numeric features for the numeric only regression model
numeric_features = weather.select_dtypes(include = ['float64']).columns.tolist()

# For numeric predictors only regression model
X_cars_num = weather[numeric_features].drop(columns = ['Apparent Temperature (C)'])

# Categorical features for the categorical only regression model
categorical_features = weather.select_dtypes(include = ['str']).columns.tolist()

# %%
# Target variable
y_w = weather['Apparent Temperature (C)']

#%%
# Question 3.3 + 3.4

# Regression model with numeric features only
X_num_train, X_num_test, yw_train, yw_test = train_test_split(X_cars_num, y_w, test_size = 0.2, random_state = 42)
wModel_num = LinearRegression().fit(X_num_train, yw_train)

yw_pred = wModel_num.predict(X_num_test)

# Mean Squared Error
mse_weatherN = mean_squared_error(yw_test, yw_pred)

# Root Mean Squared Error
rmse_weatherN = np.sqrt(mse_weatherN)
print(f"Root Mean Squared Error: {rmse_weatherN:.2f}")

# R-Squared
r_squared_weatherN = r2_score(yw_test, yw_pred)
print(f"R² Score: {r_squared_weatherN:.4f}")

# %%
# Regression model with categorical features only

# One-hot encode, with drop_first = True to avoid dummy trap
encoded_w = pd.get_dummies(weather[categorical_features], drop_first = True, prefix = ['S', 'P', 'D'])

X_encoded_train, X_encoded_test, yw2_train, yw2_test = train_test_split(encoded_w, y_w, test_size = 0.2, random_state = 42)

wModel_cat = LinearRegression().fit(X_encoded_train, yw2_train)

yw2_pred = wModel_cat.predict(X_encoded_test)

# Mean Squared Error
mse_weatherC = mean_squared_error(yw2_test, yw2_pred)

# Root Mean Squared Error
rmse_weatherC = np.sqrt(mse_weatherC)
print(f"Root Mean Squared Error: {rmse_weatherC:.2f}")

# R-Squared
r_squared_weatherC = r2_score(yw2_test, yw2_pred)
print(f"R² Score: {r_squared_weatherC:.4f}")

# %%
# Question 3.5
# The regression model with numeric features only peformed significantly better (r-squared: 0.98) than the categorical only model (r-squared: 0.5).
# This may be because temperature can often be predicted based on numeric values of contributing factors such as humidity, wind speed, etc. 
# Additionally, there were much more numeric variables than categorical features. 
 
# %%
# Question 3.6
# I learned that linear regression model fit can vary significantly depending on the features you feed it. 
# It is crucial to remove outliers, and "noise," in order to create the best prediction model.

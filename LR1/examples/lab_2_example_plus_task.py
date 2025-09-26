#%% md
# # Example of the solution the regression problem
# # Laptop price prediction
# [source of the data + code of example](https://www.kaggle.com/datasets/anubhavgoyal10/laptop-prices-dataset/data)
#%% md
# ## 0. Import the dependencies
#%%
import os

import numpy as np # linear algebra
import pandas as pd 

import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

import warnings
warnings.filterwarnings('ignore')
#%%
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn import metrics
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.linear_model import BayesianRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn import tree

from sklearn.model_selection import GridSearchCV
#%% md
# ## 1. Load the data
#%%
df = pd.read_csv("./src/laptop_price.csv")
df.head()
#%% md
# ## 2. Preprocess the data
#%%
df.columns
#%%
# select few feature to predict the price of laptop
selected_columns = ['brand', 'processor_brand', 'processor_name', 'processor_gnrtn',
                    'ram_gb', 'ram_type', 'ssd', 'hdd', 'os', 'os_bit', 'graphic_card_gb',
                    'weight', 'warranty', 'Touchscreen', 'Price']
df_train = df[selected_columns]
#%%
# check Null values in data column
print(df_train.isnull().sum())
#%%
# checking for duplicated rows
df_train.duplicated().sum()
#%%
# check overview of columns, their data types, and non-null counts.
df_train.info()
#%%
# make to different variables for categorical and numerical feature identification
cat_val = df_train.select_dtypes(include=['object']).columns
num_val = df_train.select_dtypes(include=['int', "float"]).columns

print(cat_val,'\n',num_val)
#%%
# check all unique values in features
for column in df_train.columns:
    unique_values = df_train[column].unique()
    print(f"Unique values in column '{column}': {unique_values}")
#%%
# Removed units ('GB', 'years', 'bit', etc.) from certain columns using string manipulation (str.replace) 
# and transformed them into appropriate numeric types (astype or pd.to_numeric)
df_train['ram_gb'] = df_train['ram_gb'].str.replace('GB','')
df_train['ssd'] = df_train['ssd'].str.replace('GB','')
df_train['hdd'] = df_train['hdd'].str.replace('GB','')
df_train['graphic_card_gb'] = df_train['graphic_card_gb'].str.replace('GB','')
df_train['os_bit'] = df_train['os_bit'].str.replace('-bit','')
df_train['warranty'] = df_train['warranty'].str.replace(r'\byears?\b', '', regex=True)
df_train['processor_gnrtn'] = df_train['processor_gnrtn'].str.replace('th','')


df_train['ram_gb'] = df_train['ram_gb'].astype('int32')
df_train['ssd'] = df_train['ssd'].astype('int32')
df_train['hdd'] = df_train['hdd'].astype('int32')
df_train['graphic_card_gb'] = df_train['graphic_card_gb'].astype('int32')
df_train['os_bit'] = df_train['os_bit'].astype('int32')

df_train.head()
#%%
# Convert to numeric, 'Not Available' becomes NaN
df_train['processor_gnrtn'] = pd.to_numeric(df_train['processor_gnrtn'], errors='coerce')  # Convert to numeric, 'Not Available' becomes NaN
median_value = df_train['processor_gnrtn'].median()
#%%
df_train['processor_gnrtn'].fillna(median_value, inplace=True)
df_train['processor_gnrtn'] = df_train['processor_gnrtn'].astype(int)
#%%
df_train.head()
#%%
df_train.info()
#%%
df_train.isnull().sum()
#%% md
# ## 3. Do data visualization
#%%
sns.displot(df['Price'],color='blue')
#%%
def dataplot(col):
    plt.figure(figsize= (10,6))
    sns.countplot(data = df_train, x=col, palette = 'plasma')
    plt.xticks(rotation = 'vertical')
    plt.show()
    
features = ['brand', 'ram_gb', 'processor_name', 'processor_gnrtn', 'os']

for col in features:
    dataplot(col)
#%%
plt.figure(figsize=(15,7))
sns.barplot(x = df_train['brand'], y=df_train['Price'])
plt.xticks(rotation = 'vertical')
plt.show()
#%% md
# ### Visualize the Touchscreen feature
#%%
sns.countplot(df_train, x =df_train['Touchscreen'],palette='plasma')
#%%
sns.barplot(x = df_train['Touchscreen'], y= df_train['Price'])
#%% md
# ### Visualize the Warranty feature
#%%
sns.countplot(df_train, x =df_train['warranty'],palette='plasma')
#%%
sns.barplot(x = df_train['warranty'], y= df_train['Price'])
#%% md
# ### Visualize the Weight feature
#%%
sns.countplot(df_train, x =df_train['weight'],palette='plasma')
#%%
sns.barplot(x = df_train['weight'], y= df_train['Price'])
#%% md
# ### Visualize the RAM
#%%
sns.countplot(df_train, x =df_train['ram_gb'],palette='plasma')
#%%
sns.barplot(x = df_train['ram_gb'], y= df_train['Price'])
#%%
df_train.sample(10)
#%% md
# ### Visualize the Operating_system
#%%
sns.countplot(df_train, x =df_train['os'],palette='plasma')
#%%
sns.barplot(x = df_train['os'], y= df_train['Price'])
#%% md
# ### Visualize the Price
#%%
sns.distplot(df_train['Price'])
#%%
sns.distplot(np.log(df_train['Price']))
#%%
df_train.info()
#%%
numeric_df = df_train.select_dtypes(include=['number'])
sns.heatmap(numeric_df.corr(), annot=True)
#%% md
# ## 4. Final data preps
#%%
x = df_train.drop(['Price'], axis=1)
y = np.log(df_train['Price'])
#%%
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=8418)
X_train.shape, X_test.shape
#%%
mapper = {i:value for i,value in enumerate(X_train.columns)}
mapper
#%%
df_train.head()
#%%
df_train.info()
#%% md
# ## 5.1 Test Linear Regression method
#%%
encoding = ColumnTransformer(transformers = [
    ('col_tnf', OneHotEncoder(sparse_output=False, drop='first'), [0, 1, 2, 5, 8, 11, 12, 13])
], remainder='passthrough')

lr = LinearRegression()

pipe = Pipeline([
    ('encoding', encoding),
    ('lr', lr)
])

pipe.fit(X_train, y_train)

y_train_pred = pipe.predict(X_train)
y_pred = pipe.predict(X_test)

print('Train R2 score', metrics.r2_score(y_train,y_train_pred))
print('Train MAE', metrics.mean_absolute_error(y_train,y_train_pred))
print("Train MAE on the orig price:", np.exp(metrics.mean_absolute_error(y_train,y_train_pred)))
print("Train RMSE on the orig price:", np.exp(metrics.root_mean_squared_error(y_train,y_train_pred)))
print("Train MSE on the orig price:", np.exp(metrics.mean_squared_error(y_train,y_train_pred)), "\n")

print('Test R2 score', metrics.r2_score(y_test,y_pred))
print('Test MAE', metrics.mean_absolute_error(y_test,y_pred))
print("Test MAE on the orig price:", np.exp(metrics.mean_absolute_error(y_test,y_pred)))
print("Test RMSE on the orig price:", np.exp(metrics.root_mean_squared_error(y_test,y_pred)))
print("Test MSE on the orig price:", np.exp(metrics.mean_squared_error(y_test,y_pred)), "\n")
#%% md
# # Bayesian Regression
#%%
encoding = ColumnTransformer(transformers = [
    ('col_tnf', OneHotEncoder(sparse_output=False, drop='first'), [0, 1, 2, 5, 8, 11, 12, 13])
], remainder='passthrough')

bayes = BayesianRidge()

pipe = Pipeline([
    ('encoding', encoding),
    ('bayes', bayes)
])

pipe.fit(X_train, y_train)

y_train_pred = pipe.predict(X_train)
y_pred = pipe.predict(X_test)

print('Train R2 score', metrics.r2_score(y_train,y_train_pred))
print('Train MAE', metrics.mean_absolute_error(y_train,y_train_pred))
print("Train MAE on the orig price:", np.exp(metrics.mean_absolute_error(y_train,y_train_pred)))
print("Train RMSE on the orig price:", np.exp(metrics.root_mean_squared_error(y_train,y_train_pred)))
print("Train MSE on the orig price:", np.exp(metrics.mean_squared_error(y_train,y_train_pred)), "\n")

print('Test R2 score', metrics.r2_score(y_test,y_pred))
print('Test MAE', metrics.mean_absolute_error(y_test,y_pred))
print("Test MAE on the orig price:", np.exp(metrics.mean_absolute_error(y_test,y_pred)))
print("Test RMSE on the orig price:", np.exp(metrics.root_mean_squared_error(y_test,y_pred)))
print("Test MSE on the orig price:", np.exp(metrics.mean_squared_error(y_test,y_pred)), "\n")
#%% md
# ## 5.2 Test Lasso Regression
#%%
encoding = ColumnTransformer(transformers = [
    ('col_tnf', OneHotEncoder(sparse_output=False, drop='first'), [0, 1, 2, 5, 8, 11, 12, 13])
], remainder='passthrough')

lasso = Lasso(alpha=0.001)

pipe = Pipeline([
    ('encoding', encoding),
    ('lasso', lasso)
])

pipe.fit(X_train, y_train)

y_train_pred = pipe.predict(X_train)
y_pred = pipe.predict(X_test)

print('Train R2 score', metrics.r2_score(y_train,y_train_pred))
print('Train MAE', metrics.mean_absolute_error(y_train,y_train_pred))
print("Train MAE on the orig price:", np.exp(metrics.mean_absolute_error(y_train,y_train_pred)))
print("Train RMSE on the orig price:", np.exp(metrics.root_mean_squared_error(y_train,y_train_pred)))
print("Train MSE on the orig price:", np.exp(metrics.mean_squared_error(y_train,y_train_pred)), "\n")

print('Test R2 score', metrics.r2_score(y_test,y_pred))
print('Test MAE', metrics.mean_absolute_error(y_test,y_pred))
print("Test MAE on the orig price:", np.exp(metrics.mean_absolute_error(y_test,y_pred)))
print("Test RMSE on the orig price:", np.exp(metrics.root_mean_squared_error(y_test,y_pred)))
print("Test MSE on the orig price:", np.exp(metrics.mean_squared_error(y_test,y_pred)), "\n")
#%% md
# ## 5.3. Decision Tree
#%%
encoding = ColumnTransformer(transformers = [
    ('col_tnf', OneHotEncoder(sparse_output=False, drop='first'), [0, 1, 2, 5, 8, 11, 12, 13])
], remainder='passthrough')

dtr = DecisionTreeRegressor(max_depth=8)

pipe = Pipeline([
    ('encoding', encoding),
    ('dtr', dtr)
])

pipe.fit(X_train, y_train)

y_train_pred = pipe.predict(X_train)
y_pred = pipe.predict(X_test)

print('Train R2 score', metrics.r2_score(y_train,y_train_pred))
print('Train MAE', metrics.mean_absolute_error(y_train,y_train_pred))
print("Train MAE on the orig price:", np.exp(metrics.mean_absolute_error(y_train,y_train_pred)), "\n")

print('Test R2 score', metrics.r2_score(y_test,y_pred))
print('Test MAE', metrics.mean_absolute_error(y_test,y_pred))
print("Test MAE on the orig price:", np.exp(metrics.mean_absolute_error(y_test,y_pred)))
#%% md
# ## 5.4. Random Forest
#%%
encoding = ColumnTransformer(transformers = [
    ('col_tnf', OneHotEncoder(sparse_output=False, drop='first'), [0, 1, 2, 5, 8, 11, 12, 13])
], remainder='passthrough')

random = RandomForestRegressor(n_estimators=100,
                              random_state=3,
                              max_samples=0.5,
                              max_features=0.75,
                              max_depth=15)

pipe = Pipeline([
    ('encoding', encoding),
    ('random', random)
])

pipe.fit(X_train, y_train)

y_train_pred = pipe.predict(X_train)
y_pred = pipe.predict(X_test)

print('Train R2 score', metrics.r2_score(y_train,y_train_pred))
print('Train MAE', metrics.mean_absolute_error(y_train,y_train_pred))
print("Train MAE on the orig price:", np.exp(metrics.mean_absolute_error(y_train,y_train_pred)), "\n")

print('Test R2 score', metrics.r2_score(y_test,y_pred))
print('Test MAE', metrics.mean_absolute_error(y_test,y_pred))
print("Test MAE on the orig price:", np.exp(metrics.mean_absolute_error(y_test,y_pred)))
#%% md
# ## 5.4.1 You can also tune hyperparameters of Random Forest
#%%
indexlist = [0, 1, 2, 5, 8, 11, 12, 13]
transformlist = []
for key,value in mapper.items():
    if key in indexlist:
        transformlist.append(value)
        
transformlist
#%%
x = pd.get_dummies(x,columns=transformlist,drop_first=True)
x.head()
#%%
X_train, X_test, y_train, y_test = train_test_split(x,y,
                                                   test_size=0.15,random_state=4818)

X_train.shape,X_test.shape
#%%
rfr = RandomForestRegressor()
#%%
param_grid = { 
            "n_estimators"      : [10,20,30],
            "max_features"      : ["sqrt", "log2"],
            "min_samples_split" : [2,4,8],
            "bootstrap": [True, False],
            }
#%%
grid_search = GridSearchCV(rfr, param_grid, cv=5, scoring='r2')
#%%
grid_search.fit(X_train,y_train)
#%%
print('Best hyper parameter :' , grid_search.best_params_)
print('Best model :' , grid_search.best_estimator_)
#%%
rfr =  RandomForestRegressor(bootstrap=False, max_features='log2', min_samples_split=4,
                      n_estimators=30)
#%%
rf_grid = rfr.fit(X_train,y_train)
#%%
y_train_pred = rf_grid.predict(X_train)
y_pred = rf_grid.predict(X_test)

print('Train R2 score', metrics.r2_score(y_train,y_train_pred))
print('Train MAE', metrics.mean_absolute_error(y_train,y_train_pred))
print("Train MAE on the orig price:", np.exp(metrics.mean_absolute_error(y_train,y_train_pred)), "\n")

print('Test R2 score', metrics.r2_score(y_test,y_pred))
print('Test MAE', metrics.mean_absolute_error(y_test,y_pred))
print("Test MAE on the orig price:", np.exp(metrics.mean_absolute_error(y_test,y_pred)))
#%%
predicted = []
testtrain = np.array(x)
for i in range(len(testtrain)):
    predicted.append(rfr.predict([testtrain[i]]))
    
predicted
#%%
ans = [np.exp(predicted[i][0]) for i in range(len(predicted))]
#%%
rounded_prices = [round(pred) for pred in ans]
#%%
df_train['Predicted Price'] = np.array(rounded_prices)
df_train
#%%
sns.distplot(df_train['Price'],hist=False,color='orange',label='Actual')
sns.distplot(df_train['Predicted Price'],hist=False,color='blue',label='Predicted')
plt.legend()
plt.show()
#%%
sns.scatterplot(df_train, x="Price", y="Predicted Price")
#%%
df_train["log(Price)"] = np.log(df_train["Price"])
df_train["log(Pred_Price)"] = np.log(df_train["Predicted Price"])
#%%
sns.scatterplot(df_train, x="log(Price)", y="log(Pred_Price)")
#%%

#%%

#%% md
# # The lab work task
# Solve the regression problem for the one of the presented datasets using the same steps into your work.
# To get you variant: (your_num % 9), where your_num is your number in the group list.
# 
# In addition to presented in the example methods consider other ML methods like Support Vector Machine, Gradient Boosting Regression, Bayesian Ridge Regression.
# 
# Present your work in the Jupyter Notebook variant
# 
# P.s. in provided data sources you can find examples of solving this problem, but be ready to explain the code and results.
#%% md
# # Data to choose
# 0. [Gold Price Prediction](https://www.kaggle.com/datasets/franciscogcc/financial-data)
# 1. [Possum Regression](https://www.kaggle.com/datasets/abrambeyer/openintro-possum/data)
# 2. [Student Performance](https://www.kaggle.com/datasets/nikhil7280/student-performance-multiple-linear-regression)
# 3. [Boston House Prices](https://www.kaggle.com/datasets/fedesoriano/the-boston-houseprice-data/data)
# 4. [Car Price](https://www.kaggle.com/datasets/hellbuoy/car-price-prediction)
# 5. [Medical Cost](https://www.kaggle.com/datasets/mirichoi0218/insurance)
# 6. [Crab Age Prediction](https://www.kaggle.com/datasets/sidhus/crab-age-prediction)
# 7. [Calculate Concrete Strength](https://www.kaggle.com/datasets/prathamtripathi/regression-with-neural-networking)
# 8. [Advertising dataset](https://www.kaggle.com/datasets/tawfikelmetwally/advertising-dataset)
#%%

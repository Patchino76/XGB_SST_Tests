# %% IMPORTS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import sqlite3 as db
from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn import metrics
from xgboost import XGBRegressor

 # %%
df = pd.read_csv('C:/DataSets/MFC/Features/Pulse_1T_Mill02.csv', 
                             parse_dates=True, index_col='TimeStamp')

df = df[df.index > '2022-01-01 00:00:00']
# df = df[df.Cidra200 > 15]

# for column in df.columns:
#     df[column] = df[column].ewm(span = 300).mean()
print(df.head())
# %%
feature_names = ['Grano', 'Daiki',  'Shisti',  'Class_15', 'Class_12']
feature_names = {'Ore', 'WaterMill', 'WaterZumpf', 'LevelZumpf', 'PressureHC',
                'DensityHC', 'Power', 'PulpHC', 'PumpRPM', 'MotorAmp', 
                'Grano', 'Daiki', 'Shisti', 'Class_15', 'Class_12'}

feature_names = ['Ore', 'WaterZumpf', 'WaterMill', 'PressureHC', 'MotorAmp',
                'DensityHC', 'PulpHC', 
                'Grano', 'Daiki', 'Shisti'
                ]
target_name = 'Cidra200'

X = df[feature_names]
y = df[target_name]

covariance_matrix = df[feature_names].cov()
print(covariance_matrix)
# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=12, shuffle=False)

#SCALING....
scaler_x = RobustScaler()
scaler_x.fit(X_train)
X_train = scaler_x.transform(X_train)
X_test = scaler_x.transform(X_test)

scaler_y = RobustScaler()
y_train = y_train.to_numpy().reshape(-1, 1)
scaler_y.fit(y_train)
y_train = scaler_y.transform(y_train)
y_train = y_train.reshape(-1)

y_test = y_test.to_numpy().reshape(-1, 1)
y_test = scaler_y.transform(y_test)
y_test = y_test.reshape(-1)


# %%
xgb = XGBRegressor(n_estimators=20)
# xgb = XGBRegressor(n_estimators=20, max_depth=100, eta=0.1, subsample=0.1, colsample_bytree=0.7)
xgb.fit(X_train, y_train)
print(xgb.feature_importances_)
# %%
plt.bar(range(len(xgb.feature_importances_)), xgb.feature_importances_)
# plt.show()
# %%
# Predict the model
y_train_pred = xgb.predict(X_train)
print('----------------------------XGBRegressor---------------------------')
print('R2 of Train: ', metrics.r2_score(y_train,y_train_pred))
print('Mean absolute Error: ',metrics.mean_absolute_error(y_train,y_train_pred))
print('Mean square Error: ',metrics.mean_squared_error(y_train,y_train_pred))
print('RMSE: ',np.sqrt(metrics.mean_squared_error(y_train,y_train_pred)))
print('Correlation Test: ', np.corrcoef(x = np.array(y_train), y = y_train_pred)[0,1])
print('----------------------------------------------------------------------------')

y_test_pred=xgb.predict(X_test)

print('R2 of Test: ', metrics.r2_score(y_test,y_test_pred))
print('Mean absolute Error: ',metrics.mean_absolute_error(y_test,y_test_pred))
print('Mean square Error: ',metrics.mean_squared_error(y_test,y_test_pred))
print('RMSE: ',np.sqrt(metrics.mean_squared_error(y_test,y_test_pred)))
print('Correlation Test: ', np.corrcoef(x = np.array(y_test), y = y_test_pred)[0,1])


plt.plot(np.array(y_test), 'r')
plt.plot(np.array(y_test_pred), 'k')
plt.show()
# %%
plt.scatter(np.array(y_test), np.array(y_test_pred))
plt.show()
# %%
plt.scatter(np.array(y_train), np.array(y_train_pred))
plt.show()
# %%

# %%

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

import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

 # %%
df = pd.read_csv('C:/DataSets/MFC/Features/Pulse_1T_Mill06.csv', 
                             parse_dates=True, index_col='TimeStamp')

df = df[df.index > '2022-03-01 00:00:00']
# df = df.reset_index()
# df = df.drop(['TimeStamp'], axis=1)
print(df.tail())

# for column in df.columns:
#     df[column] = df[column].ewm(span = 30).mean()

scaler = RobustScaler()
sc = scaler.fit_transform(df)
df_sc = pd.DataFrame(sc, columns=df.columns)


#%% SCALING DF

feature_names = ['Ore', 'WaterZumpf', 'WaterMill', 'PressureHC', 'MotorAmp',
                'DensityHC', 'PulpHC', 
                'Grano', 'Daiki', 'Shisti'
                ]
target_name = 'Cidra200'

X_train_, X_test_, y_train_, y_test_ = train_test_split(df_sc[feature_names], df_sc[target_name], test_size=0.15, random_state=12, shuffle=False)
X_train = np.array(X_train_)
X_test = np.array(X_test_)
y_train = np.array(y_train_)
y_test = np.array(y_test_)

# %% Emulate df with multivariate distributions
train_all = np.column_stack((np.array(X_train), np.array(y_train)))
mu = train_all.mean(axis=0)
sigma = np.cov(train_all.T)
multivar = tfd.MultivariateNormalTriL(loc=mu, scale_tril=sigma)
sample = multivar.sample(1000000)

X_sample = sample[:,:-1].numpy()
y_sample = sample[:,-1].numpy()

# %%
xgb = XGBRegressor(n_estimators=20)
# xgb = XGBRegressor(n_estimators=20, max_depth=5, eta=0.7, subsample=0.1, colsample_bytree=0.4)
xgb.fit(X_sample, y_sample)
print(xgb.feature_importances_)

# %%
plt.barh(feature_names, xgb.feature_importances_)
plt.show()
# %%
# Predict the model
y_train_pred = xgb.predict(X_train)
print('----------------------------XGBRegressor---------------------------')
print('R2 of Train: ', metrics.r2_score(y_train,y_train_pred))
print('Mean absolute Error: ',metrics.mean_absolute_error(y_train,y_train_pred))
print('Mean square Error: ',metrics.mean_squared_error(y_train,y_train_pred))
print('RMSE: ',np.sqrt(metrics.mean_squared_error(y_train,y_train_pred)))
print('Correlation Test: ', np.corrcoef(x = np.array(y_train), y = y_train_pred)[0,1])

y_test_pred=xgb.predict(X_test)

print('R2 of Test: ', metrics.r2_score(y_test,y_test_pred))
print('Mean absolute Error: ',metrics.mean_absolute_error(y_test,y_test_pred))
print('Mean square Error: ',metrics.mean_squared_error(y_test,y_test_pred))
print('RMSE: ',np.sqrt(metrics.mean_squared_error(y_test,y_test_pred)))
print('Correlation Test: ', np.corrcoef(x = np.array(y_test), y = y_test_pred)[0,1])

plt.plot(np.array(y_train), 'r')
plt.plot(np.array(y_train_pred), 'k')
plt.show()

plt.plot(np.array(y_test), 'r')
plt.plot(np.array(y_test_pred), 'k')
plt.show()
# %%

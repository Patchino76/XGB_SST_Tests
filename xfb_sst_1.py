 # %% IMPORTS

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import sqlite3 as db
from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit
from sklearn import metrics
from xgboost import XGBRegressor

 # %%
with db.connect("C:\\MFC_Scripts\\MillingProcess\\MODELING\\SST\\db.sqlite3") as conn:
    
    df = pd.read_sql_query("SELECT * from OreQuality", conn)
    
    conn.commit()

df = df.set_index(pd.DatetimeIndex(df['TimeStamp']))    
print(df.tail())
# %%
feature_names = ['Daiki',  'Shisti',  'Class_15', 'Class_12']
target_name = 'Grano'

X = df[feature_names]
y = df[target_name]
# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=12, shuffle=False)
# %%
xgb = XGBRegressor(n_estimators=100)
xgb.fit(X_train, y_train)
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

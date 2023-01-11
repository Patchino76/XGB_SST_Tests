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
from sklearn.svm import SVR

import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
from bayes_opt import BayesianOptimization

 # %%
with db.connect("C:\\MFC_Scripts\\MillingProcess\\MODELING\\SST\\db.sqlite3") as conn:
    
    df = pd.read_sql_query("SELECT * from OreQuality", conn)
    
    conn.commit()


df = df.drop(['TimeStamp'], axis=1)

feature_names = ['Daiki',  'Shisti'] #, 'Class_15', 'Class_12'
target_name = 'Grano'

df = df[[target_name, *feature_names]]

print(df.tail())


#%% SCALING DF
scaler = StandardScaler()
sc = scaler.fit_transform(df)
df_sc = pd.DataFrame(sc, columns=df.columns)



X_train_, X_test_, y_train_, y_test_ = train_test_split(df_sc[feature_names], df_sc[target_name], test_size=0.15, random_state=12, shuffle=False)
X_train = np.array(X_train_)
X_test = np.array(X_test_)
y_train = np.array(y_train_)
y_test = np.array(y_test_)

# %% Emulate df with multivariate distributions
train_all = np.column_stack((np.array(X_train), np.array(y_train)))
mu = train_all.mean(axis=0)
cov = np.cov(train_all.T)
scale = tf.linalg.cholesky(cov)

multivar = tfd.MultivariateNormalTriL(loc=mu, scale_tril=scale)
sample = multivar.sample(100000)

X_sample = sample[:,:-1].numpy()
y_sample = sample[:,-1].numpy()

# %%
xgb = SVR(kernel='rbf', C=1e3, gamma=0.1)
xgb.fit(X_sample, y_sample)

# %%
# plt.barh(feature_names, xgb.feature_importances_)
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
#BAYES OPT

#bounds = [(lo,hi) for lo,hi in zip(df_sc[feature_names].min(), df_sc[feature_names].max())]
bounds = [(lo,hi) for lo,hi in zip(df_sc[feature_names].mean()-1, df_sc[feature_names].mean()+1)]
f_args = {key: val for key,val in zip(feature_names, bounds)}

# %%
#BAYES OPT
def regressor(**data):
    
    point_vals = [val for val in data.values()]   
    point = np.array(point_vals).reshape(1, len(data))
    pr_y = xgb.predict(point)

    return pr_y[0]

#%%
optimizer = BayesianOptimization(regressor, f_args)
optimizer.maximize(20,20)
print(optimizer.max['params'])
print(optimizer.max['target'])

# %%
bo_rez = np.empty(df_sc.shape[1])
for res in optimizer.res:
    t = np.array([*res.values()][0])
    f = np.array([*[*res.values()][1].values()])
    all = np.hstack((t,f))
    print(all)
    bo_rez = np.vstack((bo_rez, all))
# %%
#Inverse transformation
inverse = scaler.inverse_transform(bo_rez)

df_bo = pd.DataFrame(inverse, columns = [target_name, *feature_names])
print(df_bo)
# %%
df_bo.plot()
# %%

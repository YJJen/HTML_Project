# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import datetime

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

data_df=pd.read_csv('/kaggle/input/hotel-booking-demand/hotel_bookings.csv')

data_df.info()

data_df['time_concat']=pd.to_datetime(data_df['arrival_date_year'].astype(int).astype(str)  + data_df['arrival_date_month'] + data_df['arrival_date_day_of_month'].astype(int).astype(str),format='%Y%B%d')
data_df['time_sub']=data_df['time_concat']-pd.to_datetime(data_df['reservation_status_date'])
data_df['time_sub']=data_df['time_sub'].astype('timedelta64[D]').astype(float)

y=data_df['is_canceled']
data_df=data_df.drop(['is_canceled'],axis=1)

float_df=data_df.describe() 

object_feature=[v for v in data_df.columns.tolist() if v not in float_df.columns and v not in ['country','arrival_date_month','time_concat','reservation_status_date']]

float_feature=[v for v in float_df.columns]

for i in float_feature:
    data_df[i]=data_df[i].astype(float)
dict_for_objs={}
for c in object_feature:
    unique_values=data_df[c].unique()
    dict_for_objs[c]=dict(zip(unique_values,list(range(len(unique_values)))))

def conver_obj_to_index(x,c):
    dict_obj_index=dict_for_objs[c]
    return dict_obj_index.get(x,-1)

for c in object_feature:
    data_df[c]=data_df[c].apply(lambda x:conver_obj_to_index(x,c))
    data_df[c]=data_df[c].astype(float)
    print("{}conversion finished".format(c))


var_list=[]
for i in data_df.columns:
    if i in float_feature or i in object_feature:
        var_list.append(i)
print(var_list)
x=data_df[var_list]


from sklearn.model_selection import train_test_split
X_train_all,X_pred,y_train_all,y_pred=train_test_split(x,y,random_state=4321,test_size=0.2)
X_train,X_test,y_train,y_test=train_test_split(X_train_all,y_train_all,random_state=4321,test_size=0.3)


import lightgbm as lgb
lgb_train=lgb.Dataset(X_train,y_train)
lgb_test=lgb.Dataset(X_test,y_test)
params={'boosting_type':'gbdt',
        'objective':'binary',
        'metric':'binary_error',
        'learning_rate':0.06,
        'num_leaves':35,
        'max_depth':5,
        'subsample':0.5,
        'subsample_freq':1,
        'colsample-bytree':0.8,
        'min_child_samples':100,
        'min_split_gain':0.05,
        'reg_alpha':1,'reg_lambda':1}
gbm=lgb.train(params,lgb_train,num_boost_round=2000,valid_sets=[lgb_train,lgb_test],early_stopping_rounds=100,verbose_eval=100)


y_pred_=gbm.predict(X_pred,num_iteration=gbm.best_iteration)

threshold = 0.5  
result=[]
for pred in y_pred_:  
    if pred > threshold:
        result.append(1)
    else:
        result.append(0)


from sklearn.metrics import accuracy_score
print(accuracy_score(y_pred,result))        
#!/usr/bin/env python
# coding: utf-8

# ## Caricamento dati

# In[1]:


import time
#import logging
#logging.basicConfig()
#logging.getLogger().setLevel(logging.DEBUG)

start = time.time()
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt
keras = tf.keras
from tensorflow.keras.utils import to_categorical
from livelossplot import PlotLossesKerasTF
import funzioni as fz
import importlib
from collections import Counter
from tensorflow.keras.utils import to_categorical
from imblearn.under_sampling import ClusterCentroids
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from imblearn.datasets import make_imbalance
from imblearn.under_sampling import NearMiss
from imblearn.metrics import classification_report_imbalanced
import pickle
from tensorflow.keras.layers import Dense, Input, Conv1D, Dropout, MaxPooling1D, Flatten, concatenate, AveragePooling1D
from tensorflow.keras.models import Model
gpus = tf.config.experimental.list_physical_devices('GPU')
from tensorflow.keras.utils import plot_model
from sklearn import preprocessing

print(gpus)
if gpus:
  # Restrict TensorFlow to only use the second GPU
  try:
    tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)
  except RuntimeError as e:
    # Visible devices must be set at program startup
    print(e)


from tensorflow.keras.layers import Conv1D, Activation, GlobalAveragePooling1D, MaxPooling1D, AveragePooling1D, Dropout, Dense, Lambda, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

def print_count(labels):
    c=Counter(labels)
    print(sorted(c.items()))
    tot=sum(c.values())
    for item in c.items():
        print(item[0], round(item[1]/tot*100, 2), '%')


# In[2]:


data=pd.read_pickle('./data/alldata_class1o.pkl')
X=data.iloc[:,0:3250]
y=data.iloc[:,3251:]
y['class']=y.apply(np.argmax, axis=1)
print_count(y['class'])

scaler = preprocessing.MinMaxScaler()
X = pd.DataFrame(data = scaler.fit_transform(X.values.T).T, columns = X.columns)
y=y['class']


def build_park_model():
    keras.backend.clear_session()
    tf.random.set_seed(0)
    np.random.seed(0)
    filters = 80
    kernel_size=200
    subsample_length = 2
    model=None
    model = keras.models.Sequential()
    model.add(keras.layers.Conv1D(filters, kernel_size, subsample_length, padding = 'same', input_shape=(X.shape[1],1))) 
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.Dropout(0.3))
    model.add(keras.layers.AveragePooling1D(3, strides=None)) 
    model.add(Lambda(lambda v: tf.cast(tf.signal.fft(tf.cast(v,dtype=tf.complex64)),tf.float32)))
    model.add(keras.layers.Conv1D(filters*2, int(kernel_size/2), subsample_length, padding = 'same')) 
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.Dropout(0.3))
    model.add(keras.layers.AveragePooling1D(3, strides=None)) 
    model.add(keras.layers.Conv1D(filters*3, int(kernel_size/4), subsample_length, padding = 'same')) 
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.Dropout(0.3))
    model.add(keras.layers.AveragePooling1D(3, strides=None)) 
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(units=2800,activation='relu', kernel_regularizer='l2')) #Aggiunto
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(units=1400,activation='relu', kernel_regularizer='l2')) #Aggiunto
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(units=700,activation='relu',kernel_regularizer='l2'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(units=70,activation='relu',kernel_regularizer='l2'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(units=7, activation='softmax'))

    optimizer = keras.optimizers.Adam()
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=["accuracy"])
    return model


cv = StratifiedKFold(n_splits=10)
sampling_strategy_under = {0: 50000, 2: 100000, 5:50000}
sampling_strategy_over = {1: 20000, 3: 20000, 4:20000, 6:20000}

cv_results=pd.DataFrame(columns=['class', 0,1,2,3,4,5,6])

count=1
for train_index, test_index in cv.split(X, y):    
    print("TRAIN:", train_index, "TEST:", test_index)
    print("FOLD:", count, "STARTED\n")
    X_train, X_test = X.values[train_index], X.values[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    under = RandomUnderSampler(sampling_strategy=sampling_strategy_under)
    over = RandomOverSampler(sampling_strategy=sampling_strategy_over)
    if count == 1: 
        print(len(y_train))
        print(len(y_test))
        print_count(y_train)
        print_count(y_test)

    X_train, y_train = under.fit_resample(X_train, y_train)
    
    if count == 1: 
        print_count(y_train)
    
    X_train, y_train = over.fit_resample(X_train, y_train)

    if count == 1: 
        print_count(y_train)

    model=build_park_model()
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    model.fit(X_train, y_train, batch_size = 200, verbose = 1, epochs=80, shuffle=True)
    probabilities = model.predict(X_test)
    df1=pd.DataFrame(y_test)
    df2=pd.DataFrame(probabilities, index=y_test.index)
    results = pd.concat([df1,df2], axis=1)
    cv_results = pd.concat([cv_results, results])
    print("FOLD:", count, "ENDED\n")    
    count+=1
    #if count==5:break


pd.to_pickle(cv_results, 'cv_results.pkl')

cv_results["conf"] = np.nan
for index, row in cv_results.iterrows():
    temp=np.argsort(-row[[0,1,2,3,4,5,6]])
    first=row[temp[0]]
    second=row[temp[1]]
    cv_results.loc[index, 'conf']=(first-second)
pd.to_pickle(results, 'cv_results_conf.pkl')

end = time.time()

print('EXECUTION TIME IN HOURS: ', round((end - start)/3600.,2))

# ---

import time
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

print(gpus)
if gpus:
  # Restrict TensorFlow to only use the second GPU
  try:
    tf.config.experimental.set_visible_devices(gpus[1], 'GPU')
    tf.config.experimental.set_memory_growth(gpus[1], True)
  except RuntimeError as e:
    # Visible devices must be set at program startup
    print(e)

import pickle
with open("train_set.pkl", "rb") as train, open('test_set.pkl', 'rb') as test:
    train_set = pickle.load(train)
    train_labels = pickle.load(train)

    test_set = pickle.load(test)
    test_labels = pickle.load(test)

from tensorflow.keras.layers import Conv1D, Activation, GlobalAveragePooling1D, MaxPooling1D, AveragePooling1D, Dropout, Dense, Lambda, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2

def build_park_model():
    keras.backend.clear_session()
    tf.random.set_seed(0)
    np.random.seed(0)
    filters = 80
    kernel_size=200
    subsample_length = 2
    model=None
    model = keras.models.Sequential()
    model.add(keras.layers.Conv1D(filters, kernel_size, subsample_length, padding = 'same', input_shape=(train_set.shape[1],1))) 
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
    model.add(keras.layers.Dense(units=2800,activation='relu')) #Aggiunto
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(units=1400,activation='relu')) #Aggiunto
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(units=700,activation='relu'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(units=70,activation='relu'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(units=7, activation='softmax'))

    optimizer = keras.optimizers.Adam()
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=["accuracy"])
    return model


model=build_park_model()
model.summary()

train_set, val_set, train_labels, val_labels = train_test_split(train_set, train_labels, test_size = 0.20, random_state = 42, stratify = train_labels)
train_labels_1o = to_categorical(train_labels)
val_labels_1o = to_categorical(val_labels)


model=build_park_model()

plotlosses = PlotLossesKerasTF()
early_stopping=keras.callbacks.EarlyStopping(monitor='val_loss', patience=200, verbose=0, mode='min')
model_checkpoint = keras.callbacks.ModelCheckpoint("script_checkpoint.h5", monitor='val_loss', mode='min', verbose=0, save_best_only=True)

history = model.fit(train_set, train_labels_1o, batch_size = 100, verbose = 1, epochs=500, validation_data=(val_set, val_labels_1o), callbacks=[model_checkpoint, early_stopping], shuffle=True)

model = keras.models.load_model("script_checkpoint.h5")


from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

c=Counter(test_labels['class'])
tot=sum(c.values())
for item in sorted(c.items()):
    print(item[0], round(item[1]/tot*100, 2), '%')
test_labels=test_labels.drop("class", axis=1)
pred_labels = model.predict(test_set)

with open('test_probs.pickle', 'wb') as handle:
        pickle.dump([test_labels.values, pred_labels], handle, protocol=pickle.HIGHEST_PROTOCOL)



pred_labels = (pred_labels > 0.5)
preds = np.argmax(pred_labels, axis=1)
labels = np.argmax(test_labels.values, axis=1)
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

print('\nAccuracy: {:.2f}\n'.format(accuracy_score(labels, preds)))

print('Micro Precision: {:.2f}'.format(precision_score(labels, preds, average='micro')))
print('Micro Recall: {:.2f}'.format(recall_score(labels, preds, average='micro')))
print('Micro F1-score: {:.2f}\n'.format(f1_score(labels, preds, average='micro')))

print('Macro Precision: {:.2f}'.format(precision_score(labels, preds, average='macro')))
print('Macro Recall: {:.2f}'.format(recall_score(labels, preds, average='macro')))
print('Macro F1-score: {:.2f}\n'.format(f1_score(labels, preds, average='macro')))

print('Weighted Precision: {:.2f}'.format(precision_score(labels, preds, average='weighted')))
print('Weighted Recall: {:.2f}'.format(recall_score(labels, preds, average='weighted')))
print('Weighted F1-score: {:.2f}'.format(f1_score(labels, preds, average='weighted')))


test_labels['class']=test_labels.apply(np.argmax, axis=1)
print(sorted(Counter(test_labels['class']).items()))
sampling_strategy = {0: 1600, 1:1600, 2: 1600, 5: 1600, 6:1600}
under = RandomUnderSampler(random_state=42, sampling_strategy=sampling_strategy)
over = RandomOverSampler(random_state=42)
test_set = test_set.reshape(test_set.shape[0], test_set.shape[1])
test_set,test_labels = under.fit_resample(test_set, test_labels['class'])
test_set,test_labels = over.fit_resample(test_set, test_labels)
c=Counter(test_labels)
tot=sum(c.values())
for item in sorted(c.items()):
    print(item[0], round(item[1]/tot*100, 2), '%')
print(sorted(Counter(test_labels).items()))
test_set = test_set.reshape(test_set.shape[0], test_set.shape[1], 1)
#samples - timesteps - features
pred_labels = model.predict(test_set)
labels_support = test_labels.values

with open('test_probs_support.pickle', 'wb') as handle:
        pickle.dump([labels_support, pred_labels], handle, protocol=pickle.HIGHEST_PROTOCOL)
                     
pred_labels = (pred_labels > 0.5)
preds_support = np.argmax(pred_labels, axis=1)

                     
print('\nClassification Report\n')
classes = ["Ortorombic","Tetragonal","Monoclinic","Trigonal","Hexagonal","Triclinic","Cubic"]
print(classification_report(labels_support, preds_support, target_names=classes))

with open('script.pickle', 'wb') as handle:
        pickle.dump([history.history, labels, preds, labels_support, preds_support], handle, protocol=pickle.HIGHEST_PROTOCOL)

end = time.time()
print(end - start)
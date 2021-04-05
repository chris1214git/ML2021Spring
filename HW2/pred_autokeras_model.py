
import numpy as np

print('Loading data ...')

data_root='./timit_11/'
train = np.load(data_root + 'train_11.npy')
train_label = np.load(data_root + 'train_label_11.npy')
test = np.load(data_root + 'test_11.npy')

print('Size of training data: {}'.format(train.shape))
print('Size of testing data: {}'.format(test.shape))

VAL_RATIO = 0.2

percent = int(train.shape[0] * (1 - VAL_RATIO))
train_x, train_y, val_x, val_y = train[:percent], train_label[:percent], train[percent:], train_label[percent:]
print('Size of training set: {}'.format(train_x.shape))
print('Size of validation set: {}'.format(val_x.shape))


import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
import autokeras as ak

# model = load_model('testmodel.h5')
model = load_model("testmodel.h5", custom_objects=ak.CUSTOM_OBJECTS)
# print(model.summary())

from keras.utils.vis_utils import plot_model
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

train_pred = model.predict(train_x)
val_pred = model.predict(val_x)

train_pred = np.argmax(train_pred, axis=1)
val_pred = np.argmax(val_pred, axis=1)

from sklearn.metrics import accuracy_score
print('Train accuracy', accuracy_score(train_pred, train_y))
print('Valid accuracy', accuracy_score(val_pred, val_y))

test_pred = model.predict(test)
test_pred = np.argmax(test_pred, axis=1)
print(test_pred)

with open('autokeras_prediction.csv', 'w') as f:
    f.write('Id,Class\n')
    for i, y in enumerate(test_pred):
        f.write('{},{}\n'.format(i, y))

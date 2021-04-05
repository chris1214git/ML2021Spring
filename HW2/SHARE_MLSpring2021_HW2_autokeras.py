#!/usr/bin/env python
# coding: utf-8

# # **Homework 2-1 Phoneme Classification**

# ## The DARPA TIMIT Acoustic-Phonetic Continuous Speech Corpus (TIMIT)
# The TIMIT corpus of reading speech has been designed to provide speech data for the acquisition of acoustic-phonetic knowledge and for the development and evaluation of automatic speech recognition systems.
# 
# This homework is a multiclass classification task, 
# we are going to train a deep neural network classifier to predict the phonemes for each frame from the speech corpus TIMIT.
# 
# link: https://academictorrents.com/details/34e2b78745138186976cbc27939b1b34d18bd5b3

# ## Download Data
# Download data from google drive, then unzip it.
# 
# You should have `timit_11/train_11.npy`, `timit_11/train_label_11.npy`, and `timit_11/test_11.npy` after running this block.<br><br>
# `timit_11/`
# - `train_11.npy`: training data<br>
# - `train_label_11.npy`: training label<br>
# - `test_11.npy`:  testing data<br><br>
# 
# **notes: if the google drive link is dead, you can download the data directly from Kaggle and upload it to the workspace**
# 
# 
# 

# ## Preparing Data
# Load the training and testing data from the `.npy` file (NumPy array).

# In[1]:

trial = 1

import numpy as np

print('Loading data ...')

data_root='./timit_11/'
train = np.load(data_root + 'train_11.npy')
train_label = np.load(data_root + 'train_label_11.npy')
test = np.load(data_root + 'test_11.npy')

print('Size of training data: {}'.format(train.shape))
print('Size of testing data: {}'.format(test.shape))


# ## Create Dataset

# In[2]:


import torch
from torch.utils.data import Dataset

class TIMITDataset(Dataset):
    def __init__(self, X, y=None):
        self.data = torch.from_numpy(X).float()
        if y is not None:
            y = y.astype(np.int)
            self.label = torch.LongTensor(y)
        else:
            self.label = None

    def __getitem__(self, idx):
        if self.label is not None:
            return self.data[idx], self.label[idx]
        else:
            return self.data[idx]

    def __len__(self):
        return len(self.data)


# Split the labeled data into a training set and a validation set, you can modify the variable `VAL_RATIO` to change the ratio of validation data.

# In[3]:


VAL_RATIO = 0.2

percent = int(train.shape[0] * (1 - VAL_RATIO))
train_x, train_y, val_x, val_y = train[:percent], train_label[:percent], train[percent:], train_label[percent:]
print('Size of training set: {}'.format(train_x.shape))
print('Size of validation set: {}'.format(val_x.shape))


# Create a data loader from the dataset, feel free to tweak the variable `BATCH_SIZE` here.

# Cleanup the unneeded variables to save memory.<br>
# 
# **notes: if you need to use these variables later, then you may remove this block or clean up unneeded variables later<br>the data size is quite huge, so be aware of memory usage in colab**

# ## Create Model

# In[4]:


import numpy as np
import pandas as pd
import tensorflow as tf

import autokeras as ak


# In[ ]:


# Initialize the structured data classifier.
clf = ak.StructuredDataClassifier(
    overwrite=False,
    max_trials=100,
    tuner='bayesian',
    project_name='structured_data_classifier_{}'.format(trial),
    seed=133
)  
# Feed the structured data classifier with training data.
clf.fit(
    train_x,
    train_y,
    epochs=20, 
    batch_size=320,
    validation_data=(val_x, val_y),
)


best_model = clf.export_model()

# best_model = clf.tuner.get_best_model()
best_model.save("autokeras_model0321_{}.h5".format(trial))
print(best_model)

import keras
from keras.utils.vis_utils import plot_model
plot_model(best_model, to_file='model_plot_{}.png'.format(trial), show_shapes=True, show_layer_names=True)

# Predict with the best model.
train_pred = best_model.predict(train_x.astype(np.unicode))
val_pred = best_model.predict(val_x.astype(np.unicode))


print('Train accuracy', (train_pred==train_y).sum() / train_y.shape[0])
print('Valid accuracy', (val_pred==val_y).sum() / val_y.shape[0])

test_pred = best_model.predict(test.astype(np.unicode))


# Create a testing dataset, and load model from the saved checkpoint.

# In[ ]:


# # create testing dataset
# test_set = TIMITDataset(test, None)
# test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)


# Make prediction.

# In[ ]:





# Write prediction to a CSV file.
# 
# After finish running this block, download the file `prediction.csv` from the files section on the left-hand side and submit it to Kaggle.

# In[ ]:


with open('prediction_auto.csv', 'w') as f:
    f.write('Id,Class\n')
    for i, y in enumerate(test_pred):
        f.write('{},{}\n'.format(i, y))


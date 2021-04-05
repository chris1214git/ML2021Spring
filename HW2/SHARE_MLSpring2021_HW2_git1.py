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

# In[1]:


# !gdown --id '1HPkcmQmFGu-3OknddKIa5dNDsR05lIQR' --output data.zip
# !unzip data.zip
# !ls 


# ## Preparing Data
# Load the training and testing data from the `.npy` file (NumPy array).

# In[2]:


import torch
torch.cuda.set_device(1)
print('cuda:', torch.cuda.current_device())
device = "cuda" if torch.cuda.is_available() else "cpu"

import argparse
import os, sys
import json

parser = argparse.ArgumentParser()
parser.add_argument('--last_hidden_dim', type=int, default=500)
parser.add_argument('--skip_connection_layer', type=int, nargs='+', default=[0, 2])
parser.add_argument('--skip_connection_layer_deactivate', type=int, default=0)
parser.add_argument('--hidden_activation', type=str, default='lrelu')
parser.add_argument('--dropout', type=float, default=0.3)
parser.add_argument('--l1', type=float, default=0.)
parser.add_argument('--l2', type=float, default=0.)
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--batch_size', type=int, default=320)

parser.add_argument('--opt', type=str, default='adam')
parser.add_argument('--weight_decay', type=float, default=0.)
parser.add_argument('--opt_scheduler', type=str, default='none')
parser.add_argument('--opt_decay_step', type=int, default=1000)
parser.add_argument('--opt_decay_rate', type=float, default=0.9)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--train_all', type=int, default=0)
parser.add_argument('--log_dir', type=str, default='dnn0')
args = parser.parse_args()
print(args)

log_path = './record2/{}/'.format(args.log_dir)
os.makedirs(log_path)
args_dict = vars(args)
cmd_input = 'python ' + ' '.join(sys.argv) + '\n'
with open(os.path.join(log_path, 'cmd_input.txt'), 'a') as f:
    f.write(cmd_input)

with open(os.path.join(log_path, 'para.txt'), 'a') as f:
    f.write(json.dumps(args_dict))

import numpy as np

print('Loading data ...')

data_root='./timit_11/'
train = np.load(data_root + 'train_11.npy')
train_label = np.load(data_root + 'train_label_11.npy')
test = np.load(data_root + 'test_11.npy')
# test_label = np.load(data_root + 'test_label_11.npy')

print('Size of training data: {}'.format(train.shape))
print('Size of testing data: {}'.format(test.shape))


# ## Create Dataset

# In[3]:


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

# In[4]:

if args.train_all == 1:
	VAL_RATIO = 0
else:
	VAL_RATIO = 0.2

percent = int(train.shape[0] * (1 - VAL_RATIO))
train_x, train_y, val_x, val_y = train[:percent], train_label[:percent], train[percent:], train_label[percent:]
print('Size of training set: {}'.format(train_x.shape))
print('Size of validation set: {}'.format(val_x.shape))

train_test = np.concatenate((train, test), axis=0)
# train_test_label = np.concatenate((train_label, test_label), axis=0)
print('Size of all set: {}'.format(train.shape))
print('Size of train + test all set: {}'.format(train_test.shape))

# Create a data loader from the dataset, feel free to tweak the variable `BATCH_SIZE` here.

# In[5]:


BATCH_SIZE = args.batch_size

from torch.utils.data import DataLoader

train_set = TIMITDataset(train_x, train_y)
val_set = TIMITDataset(val_x, val_y)
train_all_set = TIMITDataset(train, train_label)
# train_test_all_set = TIMITDataset(train_test, train_test_label)


# if args.train_all == 1:
#     train_loader = DataLoader(train_all_set, batch_size=BATCH_SIZE, shuffle=True) #only shuffle the training data
# elif args.train_all == 2:
#     train_loader = DataLoader(train_test_all_set, batch_size=BATCH_SIZE, shuffle=True) #only shuffle the training data
# elif args.train_all == 0:
#     train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True) #only shuffle the training data
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True) #only shuffle the training data
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

print('Train num', len(train_loader.dataset))
print('Val num', len(val_loader.dataset))

# Cleanup the unneeded variables to save memory.<br>
# 
# **notes: if you need to use these variables later, then you may remove this block or clean up unneeded variables later<br>the data size is quite huge, so be aware of memory usage in colab**

# In[6]:


import gc

del train, train_label, train_x, train_y, val_x, val_y
gc.collect()


# ## Create Model

# Define model architecture, you are encouraged to change and experiment with the model architecture.


# In[14]:


import torch
import numpy as np
import torch.nn as nn

def get_activation(activation):
    if activation == 'relu':
        return torch.nn.ReLU()
    elif activation == 'lrelu':
        return torch.nn.LeakyReLU()
    elif activation == 'prelu':
        return torch.nn.PReLU()
    elif activation == 'tanh':
        return torch.nn.Tanh()
    elif activation == 'sigmoid':
        return torch.nn.Sigmoid()
    elif (activation is None) or (activation == 'none'):
        return torch.nn.Identity()
    else:
        raise NotImplementedError

class SpeechClassifier(nn.Module):
    def __init__(self, size_list, residual_blocks_idx):
        super(SpeechClassifier, self).__init__()
        layers = []
        self.size_list = size_list
        self.layers = []
        self.residual_blocks_idx = residual_blocks_idx
        for i in range(len(size_list) - 2):
            self.layers.append(nn.Linear(size_list[i],size_list[i+1]))
            self.layers.append(nn.BatchNorm1d(size_list[i+1]))
            self.layers.append(get_activation(args.hidden_activation))
            # self.layers.append(nn.LeakyReLU())
            self.layers.append(nn.Dropout(p=args.dropout))
        self.layers.append(nn.Linear(size_list[-2], size_list[-1]))
        self.n_layers = len(self.layers)
        self.layers = nn.ModuleList(self.layers)

    def forward(self, x):
        x = x.view(-1, self.size_list[0]) # Flatten the input
        out = x
        residual_x = x
        residual_idx = 0
        for i in range(self.n_layers):
            # 3 layers per linear.
            ## linear + b-norm + relu
            out = self.layers[i](out)
            ## apple skip connections at residual_blocks_idx layers.
            # if residual_idx < len(self.residual_blocks_idx):
            #     if ((self.residual_blocks_idx[residual_idx]*3)+1) == i:
            #         residual_x = out
            #     if (((self.residual_blocks_idx[residual_idx]+2)*3)+1) == i:
            #         out = out + residual_x
            #         residual_idx += 1
            if args.skip_connection_layer_deactivate == 0:
                if residual_idx < len(self.residual_blocks_idx):
                    if ((self.residual_blocks_idx[residual_idx]*4)+1) == i:
                        residual_x = out
                    if (((self.residual_blocks_idx[residual_idx]+2)*4)+1) == i:
                        out = out + residual_x
                        residual_idx += 1
        return out


# ## Training

# In[15]:


#check device
def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


# Fix random seeds for reproducibility.

# In[16]:


# fix random seed
def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  
    np.random.seed(seed)  
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# Feel free to change the training parameters here.

# In[17]:


# fix random seed for reproducibility
same_seeds(args.seed)

# get device 
device = get_device()
print(f'DEVICE: {device}')

# training parameters
num_epoch = args.epochs               # number of training epoch
learning_rate = args.lr       # learning rate

# the path where checkpoint saved
model_path = os.path.join(log_path, './model.ckpt')


# create model, define a loss function, and optimizer
# model = Classifier().to(device)
# criterion = nn.CrossEntropyLoss() 
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# In[18]:

       
# LEARNING_RATE_STEP = 0.7

model_size_list = [429, 1200, 1200, 1200, 1200, 1200, 800, args.last_hidden_dim, 39]           #Frame context = 12
# Residual block input at 750 -> 800, 800 -> 800, 800 -> 800
residual_blocks_idx = [0, 2]     # Inputs to residual blocks
residual_blocks_idx = args.skip_connection_layer     # Inputs to residual blocks

model = SpeechClassifier(model_size_list, residual_blocks_idx).to(device)
criterion = nn.CrossEntropyLoss() 

from torch import optim
def build_optimizer(args, params):
    weight_decay = args.weight_decay
    if args.opt == 'adam':
        optimizer = optim.Adam(params, lr=args.lr, weight_decay=weight_decay)
    elif args.opt == 'sgd':
        optimizer = optim.SGD(params, lr=args.lr, momentum=0.95, weight_decay=weight_decay)
    elif args.opt == 'rmsprop':
        optimizer = optim.RMSprop(params, lr=args.lr, weight_decay=weight_decay)
    elif args.opt == 'adagrad':
        optimizer = optim.Adagrad(params, lr=args.lr, weight_decay=weight_decay)
    if args.opt_scheduler == 'none':
        return None, optimizer
    elif args.opt_scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.opt_decay_step, gamma=args.opt_decay_rate)
    elif args.opt_scheduler == 'cos':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    return scheduler, optimizer

scheduler, optimizer = build_optimizer(args, model.parameters())
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)    
# scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = LEARNING_RATE_STEP)


# In[19]:


# start training

best_acc = 0.0
for epoch in range(num_epoch):
    train_acc = 0.0
    train_loss = 0.0
    val_acc = 0.0
    val_loss = 0.0

    # training
    model.train() # set the model to training mode
    for i, data in enumerate(train_loader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad() 
        outputs = model(inputs) 
        batch_loss = criterion(outputs, labels)

        all_params = torch.cat([x.view(-1) for x in model.parameters()])
        l1_regularization = args.l1 * torch.norm(all_params, 1)
        l2_regularization = args.l2 * torch.norm(all_params, 2)
        batch_loss = batch_loss + l1_regularization + l2_regularization
        
        _, train_pred = torch.max(outputs, 1) # get the index of the class with the highest probability
        batch_loss.backward() 
        optimizer.step() 

        train_acc += (train_pred.cpu() == labels.cpu()).sum().item()
        train_loss += batch_loss.item()

    if scheduler is not None:
        scheduler.step(epoch)
        
    # validation
    if len(val_set) > 0:
        model.eval() # set the model to evaluation mode
        with torch.no_grad():
            for i, data in enumerate(val_loader):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                batch_loss = criterion(outputs, labels) 
                _, val_pred = torch.max(outputs, 1) 
            
                val_acc += (val_pred.cpu() == labels.cpu()).sum().item() # get the index of the class with the highest probability
                val_loss += batch_loss.item()

            print('[{:03d}/{:03d}] Train Acc: {:3.6f} Loss: {:3.6f} | Val Acc: {:3.6f} loss: {:3.6f}'.format(
                epoch + 1, num_epoch, train_acc/len(train_loader.dataset), train_loss/len(train_loader), val_acc/len(val_set), val_loss/len(val_loader)
            ))

            # if the model improves, save a checkpoint at this epoch
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), model_path)
                print('saving model with acc {:.3f}'.format(best_acc/len(val_set)))
    else:
        print('[{:03d}/{:03d}] Train Acc: {:3.6f} Loss: {:3.6f}'.format(
            epoch + 1, num_epoch, train_acc/len(train_loader.dataset), train_loss/len(train_loader)
        ))

    # if args.train_all == 1:    	
    #     torch.save(model.state_dict(), model_path)
    #     print('Last epoch', args.epochs)
    #     print('saving model with acc {:.3f}'.format(best_acc/len(val_set)))
    
# if not validating, save the last epoch
if len(val_set) == 0:
    torch.save(model.state_dict(), model_path)
    print('saving model at last epoch')


# ## Testing

# Create a testing dataset, and load model from the saved checkpoint.

# In[21]:


# create testing dataset
test_set = TIMITDataset(test, None)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

# create model and load weights from checkpoint
model = SpeechClassifier(model_size_list, residual_blocks_idx).to(device)
model.load_state_dict(torch.load(model_path))


# Make prediction.

# In[22]:


predict = []
model.eval() # set the model to evaluation mode
with torch.no_grad():
    for i, data in enumerate(test_loader):
        inputs = data
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, test_pred = torch.max(outputs, 1) # get the index of the class with the highest probability

        for y in test_pred.cpu().numpy():
            predict.append(y)


# Write prediction to a CSV file.
# 
# After finish running this block, download the file `prediction.csv` from the files section on the left-hand side and submit it to Kaggle.

# In[23]:


with open(os.path.join(log_path, 'prediction.csv'), 'w') as f:
    f.write('Id,Class\n')
    for i, y in enumerate(predict):
        f.write('{},{}\n'.format(i, y))


# In[ ]:





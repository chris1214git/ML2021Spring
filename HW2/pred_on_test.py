import argparse
import os, sys
import json
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--last_hidden_dim', type=int, default=500)
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

path = 'record2'
path2 = 'dnn7'
def load_model_path(path='record2', path2='dnn'):
    model_path_list = []

    dir_paths = os.listdir(path)
    for d in dir_paths:
        if path2 in d:
            model_path_list.append(os.path.join(path,d,'model.ckpt'))
    print(model_path_list)
    return model_path_list

model_path_list = load_model_path(path=path, path2=path2)

print('Loading data ...')

data_root='./timit_11/'
train = np.load(data_root + 'train_11.npy')
train_label = np.load(data_root + 'train_label_11.npy')
test = np.load(data_root + 'test_11.npy')

print('Size of training data: {}'.format(train.shape))
print('Size of testing data: {}'.format(test.shape))


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


BATCH_SIZE = 2048

from torch.utils.data import DataLoader
import torch
import numpy as np
import torch.nn as nn

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
            self.layers.append(nn.LeakyReLU())
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
            if residual_idx < len(self.residual_blocks_idx):
                if ((self.residual_blocks_idx[residual_idx]*3)+1) == i:
                    residual_x = out
                if (((self.residual_blocks_idx[residual_idx]+2)*3)+1) == i:
                    out = out + residual_x
                    residual_idx += 1
        return out


# ## Training

# In[15]:


#check device
def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

device = get_device()
# create testing dataset
test_set = TIMITDataset(test, None)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)


def ensemble_vote(model_path_list):
    predicts = []
    for model_path in model_path_list:
        print(model_path)
        # create model and load weights from checkpoint
        model_size_list = [429, 1200, 1200, 1200, 1200, 1200, 800, 600, 39]           #Frame context = 12
        # Residual block input at 750 -> 800, 800 -> 800, 800 -> 800
        residual_blocks_idx = [0, 2]     # Inputs to residual blocks
        model = SpeechClassifier(model_size_list, residual_blocks_idx).to(device)
        model.load_state_dict(torch.load(model_path))

        predict = []
        model.eval() # set the model to evaluation mode
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                inputs = data
                inputs = inputs.to(device)
                outputs = model(inputs)
                _, test_pred = torch.max(outputs, 1)

                for y in test_pred.cpu().numpy():
                    predict.append(y)
        predicts.append(predict)

    predicts = np.array(predicts)
    from scipy.stats import mode

    print(predicts.shape)
    predicts, _ = mode(predicts, axis=0)    
    print(predicts.shape)
    predicts = predicts.flatten()

    return predicts


def ensemble_avg(model_path_list):
    predicts = []
    for model_path in model_path_list:
        # create model and load weights from checkpoint
        model_size_list = [429, 1200, 1200, 1200, 1200, 1200, 800, 600, 39]           #Frame context = 12
        # Residual block input at 750 -> 800, 800 -> 800, 800 -> 800
        residual_blocks_idx = [0, 2]     # Inputs to residual blocks
        model = SpeechClassifier(model_size_list, residual_blocks_idx).to(device)
        model.load_state_dict(torch.load(model_path))

        predict = []
        model.eval() # set the model to evaluation mode
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                inputs = data
                inputs = inputs.to(device)
                outputs = model(inputs)
                # _, test_pred = torch.max(outputs, 1)

                for y in outputs.cpu().numpy():
                    predict.append(y)
        
        predict = np.array(predict)
        predicts.append(predict)

    predicts = np.array(predicts)
    print(predicts.shape)
    predicts = np.mean(predicts, axis=0)    
    print(predicts.shape)
    predicts = np.argmax(predicts, axis=1)    
    print(predicts.shape)    
    return predicts

# predicts = ensemble_vote(model_path_list)
predicts = ensemble_avg(model_path_list)
np.save('timit_11/test_label_11.npy', predicts)

with open(os.path.join('prediction_ensemble_avg_{}_{}.csv'.format(path, path2)), 'w') as f:
    f.write('Id,Class\n')
    for i, y in enumerate(predicts):
        f.write('{},{}\n'.format(i, y))


# In[ ]:





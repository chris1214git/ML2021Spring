import argparse
import os, sys
import json

parser = argparse.ArgumentParser()
parser.add_argument('--model_types', type=str, default='DNN')
parser.add_argument('--hidden_dims', type=int, nargs='+', default=[1024, 1024, 1024, 1024])
parser.add_argument('--hidden_activation', type=str, default='relu')
parser.add_argument('--dropout', type=float, default=0.)
parser.add_argument('--batch_norm', action='store_true')
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

parser.add_argument('--log_dir', type=str, default='dnn0')
args = parser.parse_args()
print(args)


log_path = './record/{}/'.format(args.log_dir)
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


VAL_RATIO = 0.2

percent = int(train.shape[0] * (1 - VAL_RATIO))
train_x, train_y, val_x, val_y = train[:percent], train_label[:percent], train[percent:], train_label[percent:]
print('Size of training set: {}'.format(train_x.shape))
print('Size of validation set: {}'.format(val_x.shape))

BATCH_SIZE = args.batch_size

from torch.utils.data import DataLoader

train_set = TIMITDataset(train_x, train_y)
val_set = TIMITDataset(val_x, val_y)
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True) #only shuffle the training data
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

import gc
del train, train_label, train_x, train_y, val_x, val_y
gc.collect()

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

def get_activation(activation):
    if activation == 'relu':
        return torch.nn.ReLU()
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
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

class DNN(torch.nn.Module):
    def __init__(self, 
                hidden_layer_sizes=(64,),
                hidden_activation='relu',
                batch_norm=True,
                dropout=0.):
        super(DNN, self).__init__()

        layers = nn.ModuleList()

        input_dim = 429
        output_dim = 39
        
        for layer_size in hidden_layer_sizes:
            hidden_dim = layer_size
            if batch_norm:
                layer = nn.Sequential(
                        nn.Linear(input_dim, hidden_dim),
                        nn.BatchNorm1d(num_features=hidden_dim),
                        get_activation(hidden_activation),
                        nn.Dropout(dropout),
                        )
            else:
                layer = nn.Sequential(
                        nn.Linear(input_dim, hidden_dim),
                        get_activation(hidden_activation),
                        nn.Dropout(dropout),
                        )

            layers.append(layer)
            input_dim = hidden_dim

        layer = nn.Sequential(
                        nn.Linear(input_dim, output_dim),
                        nn.Softmax(dim=1)
                        )
        layers.append(layer)
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

#check device
def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

# fix random seed
def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  
    np.random.seed(seed)  
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# fix random seed for reproducibility
same_seeds(0)

# get device 
device = get_device()
print(f'DEVICE: {device}')

# training parameters
num_epoch = args.epochs               # number of training epoch
learning_rate = args.lr       # learning rate

# the path where checkpoint saved
model_path = os.path.join(log_path, './model.ckpt')

# DNN setting
hidden_dims = args.hidden_dims
activation = args.hidden_activation
batch_norm = args.batch_norm
dropout = args.dropout

# create model, define a loss function, and optimizer
model = DNN(hidden_dims, activation, batch_norm, dropout).to(device)

criterion = nn.CrossEntropyLoss() 

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
print(model)

# start training
from tqdm import tqdm

best_acc = 0.0
Lr = []

for epoch in range(num_epoch):
    train_acc = 0.0
    train_loss = 0.0
    val_acc = 0.0
    val_loss = 0.0

    # training
    model.train() # set the model to training mode
    for i, data in enumerate(tqdm(train_loader)):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad() 
        outputs = model(inputs) 
        batch_loss = criterion(outputs, labels)
        all_params = torch.cat([x.view(-1) for x in model.parameters()])
        l1_regularization = args.l1 * torch.norm(all_params, 1)
        l2_regularization = args.l2 * torch.norm(all_params, 2)
        # batch_loss = batch_loss + l1_regularization + l2_regularization
        
        _, train_pred = torch.max(outputs, 1) # get the index of the class with the highest probability
        batch_loss.backward() 
        optimizer.step() 

        train_acc += (train_pred.cpu() == labels.cpu()).sum().item()
        train_loss += batch_loss.item()

    if scheduler is not None:
        scheduler.step(epoch)

    for param_group in optimizer.param_groups:
        Lr.append(param_group['lr'])    

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
                epoch + 1, num_epoch, train_acc/len(train_set), train_loss/len(train_loader), val_acc/len(val_set), val_loss/len(val_loader)
            ))

            # if the model improves, save a checkpoint at this epoch
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), model_path)
                print('saving model with acc {:.3f}'.format(best_acc/len(val_set)))
    else:
        print('[{:03d}/{:03d}] Train Acc: {:3.6f} Loss: {:3.6f}'.format(
            epoch + 1, num_epoch, train_acc/len(train_set), train_loss/len(train_loader)
        ))

# if not validating, save the last epoch
if len(val_set) == 0:
    torch.save(model.state_dict(), model_path)
    print('saving model at last epoch')

# create testing dataset
test_set = TIMITDataset(test, None)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

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

with open(os.path.join(log_path, 'prediction.csv'), 'w') as f:
    f.write('Id,Class\n')
    for i, y in enumerate(predict):
        f.write('{},{}\n'.format(i, y))



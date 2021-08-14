#!/usr/bin/env python
# coding: utf-8

# Homework 13 - Network Compression
# ===
# 
# > Author: Arvin Liu (r09922071@ntu.edu.tw), this colab is modified from ML2021-HW3
# 
# If you have any questions, feel free to ask: ntu-ml-2021spring-ta@googlegroups.com

# ## **Intro**
# 
# HW13 is about network compression
# 
# There are many types of Network/Model Compression,  here we introduce two:
# * Knowledge Distillation
# * Design Architecture
# 
# 
# The process of this notebook is as follows: <br/>
# 1. Introduce depthwise, pointwise and group convolution in MobileNet.
# 2. Design the model of this colab
# 3. Introduce Knowledge-Distillation
# 4. Set up TeacherNet and it would be helpful in training
# 

# ## **About the Dataset**  *(same as HW3)*
# 
# The dataset used here is food-11, a collection of food images in 11 classes.
# 
# For the requirement in the homework, TAs slightly modified the data.
# Please DO NOT access the original fully-labeled training data or testing labels.
# 
# Also, the modified dataset is for this course only, and any further distribution or commercial use is forbidden.

# In[1]:

import os
import argparse
import sys
import json

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='ghost')
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--batch_size', type=int, default=64)

# parser.add_argument('--augment', type=str, default=None)
parser.add_argument('--augment', type=int, default=0)
parser.add_argument('--img_size', type=int, default=142)
parser.add_argument('--h_dim', type=int, default=32)

parser.add_argument('--alpha', type=float, default=0.5)
parser.add_argument('--T', type=float, default=10)
parser.add_argument('--teacher', type=str, default=None)
parser.add_argument('--aux_clf', type=int, default=0)


parser.add_argument('--opt', type=str, default='adam', choices=['sgd', 'adam', 'adamw', 'adamp'])
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--weight_decay', type=float, default=1e-5)
parser.add_argument('--momentum', type=float, default=0.9)

parser.add_argument('--opt_scheduler', type=str, default='none')
parser.add_argument('--opt_scheduler_tinit', type=int, default=50)
parser.add_argument('--opt_scheduler_cycle', type=int, default=1)
parser.add_argument('--opt_scheduler_warmup_t', type=int, default=10)
parser.add_argument('--opt_scheduler_minlr', type=float, default=0.0003)
# https://fastai.github.io/timmdocs/SGDR#cycle_limit
parser.add_argument('--resume', type=str, default=None)
parser.add_argument('--modified_test', action="store_true")
parser.add_argument('--origin_opt', action="store_true")
parser.add_argument('--crit', type=str, default='ce', choices=['ce', 'ls'])
parser.add_argument('--ls_smooth', type=float, default=0.2)


parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--do_semi', type=int, default=1)
parser.add_argument('--semi_batch_size', type=int, default=32)
parser.add_argument('--semi_epoch', type=int, default=50)
parser.add_argument('--semi_init_epoch', type=float, default=300)
parser.add_argument('--threshold', type=float, default=0.5)
parser.add_argument('--select_num', type=int, default=1000)

parser.add_argument('--cuda', type=int, default=0)
parser.add_argument('--log_dir', type=str, default='0')
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

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda)


# In[2]:


# ### This block is same as HW3 ###
# # Download the dataset
# # You may choose where to download the data.

# # Google Drive
# !gdown --id '1awF7pZ9Dz7X1jn1_QAiKN-_v56veCEKy' --output food-11.zip
# # If you cannot successfully gdown, you can change a link. (Backup link is provided at the bottom of this colab tutorial).

# # Dropbox
# # !wget https://www.dropbox.com/s/m9q6273jl3djall/food-11.zip -O food-11.zip

# # MEGA
# # !sudo apt install megatools
# # !megadl "https://mega.nz/#!zt1TTIhK!ZuMbg5ZjGWzWX1I6nEUbfjMZgCmAgeqJlwDkqdIryfg"

# # Unzip the dataset.
# # This may take some time.
# !unzip -q food-11.zip


# ## **Import Packages**  *(same as HW3)*
# 
# First, we need to import packages that will be used later.
# 
# In this homework, we highly rely on **torchvision**, a library of PyTorch.

# In[3]:


### This block is same as HW3 ###
# Import necessary packages.
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision.transforms as transforms
import torchvision.models as models

from PIL import Image
# "ConcatDataset" and "Subset" are possibly useful when doing semi-supervised learning.
from torch.utils.data import ConcatDataset, DataLoader, Subset
from torchvision.datasets import DatasetFolder

# This is for the progress bar.
# from tqdm.auto import tqdm
from tqdm import tqdm
import pandas as pd


# ## **Dataset, Data Loader, and Transforms** *(similar to HW3)*
# 
# Torchvision provides lots of useful utilities for image preprocessing, data wrapping as well as data augmentation.
# 
# Here, since our data are stored in folders by class labels, we can directly apply **torchvision.datasets.DatasetFolder** for wrapping data without much effort.
# 
# Please refer to [PyTorch official website](https://pytorch.org/vision/stable/transforms.html) for details about different transforms.
# 
# ---
# **The only diffference with HW3 is that the transform functions are different.**

# In[4]:


### This block is similar to HW3 ###
# It is important to do data augmentation in training.
# However, not every augmentation is useful.
# Please think about what kind of augmentation is helpful for food recognition.
img_size = args.img_size

if args.augment == 0:
	train_tfm = transforms.Compose([
	  # Resize the image into a fixed shape (height = width = 142)
		transforms.Resize((img_size, img_size)),
	  transforms.RandomHorizontalFlip(),
	  transforms.RandomRotation(15),
		transforms.RandomCrop(img_size),
		transforms.ToTensor(),
	])

	# We don't need augmentations in testing and validation.
	# All we need here is to resize the PIL image and transform it into Tensor.
	test_tfm = transforms.Compose([
	    # Resize the image into a fixed shape (height = width = 142)
	    transforms.Resize((img_size, img_size)),
	    transforms.CenterCrop(img_size),
	    transforms.ToTensor(),
	])
elif args.augment == 1:
	train_tfm = transforms.Compose([
	    # Resize the image into a fixed shape (height = width = 128)
	    transforms.RandomResizedCrop((img_size, img_size)),

	    # You may add some transforms here.
	    # ToTensor() should be the last one of the transforms.
	    transforms.RandomAffine(degrees=10, translate=(0, 0.2), scale=(0.9, 1.1), shear=(6, 9)),
	    transforms.RandomHorizontalFlip(p=0.5),
	    transforms.ToTensor(),
	])

	# We don't need augmentations in testing and validation.
	# All we need here is to resize the PIL image and transform it into Tensor.
	test_tfm = transforms.Compose([
	    transforms.Resize((img_size, img_size)),
	    transforms.ToTensor(),
	])


# In[6]:


### This block is similar to HW3 ###
# Batch size for training, validation, and testing.
# A greater batch size usually gives a more stable gradient.
# But the GPU memory is limited, so please adjust it carefully.
batch_size = args.batch_size

# Construct datasets.
# The argument "loader" tells how torchvision reads the data.
train_set = DatasetFolder("food-11/training/labeled", loader=lambda x: Image.open(x), extensions="jpg", transform=train_tfm)
valid_set = DatasetFolder("food-11/validation", loader=lambda x: Image.open(x), extensions="jpg", transform=test_tfm)
unlabeled_set = DatasetFolder("food-11/training/unlabeled", loader=lambda x: Image.open(x), extensions="jpg", transform=train_tfm)
test_set = DatasetFolder("food-11/testing", loader=lambda x: Image.open(x), extensions="jpg", transform=test_tfm)

# Construct data loaders.
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)


# # **Architecture / Model Design**
# The following are types of convolution layer design that has fewer parameters.
# 
# ## **Depthwise & Pointwise Convolution**
# ![](https://i.imgur.com/FBgcA0s.png)
# > Blue: the connection between layers \
# > Green: the expansion of **receptive field** \
# > (reference: arxiv:1810.04231)
# 
# (a) normal convolution layer: It is fully connected. The difference between fully connected layer and fully connected convolution layer is the operation. (multiply --> convolution)
# 
# (b) Depthwise convolution layer(DW): You can consider each feature map pass through their own filter and then pass through pointwise convolution layer(PW) to combine the information of all pixels in feature maps.
# 
# 
# (c) Group convolution layer(GC): Group the feature maps. Each group passes their filter then concate together. If group_size = input_feature_size, then GC becomes DC (channels are independent). If group_size = 1, then GC becomes fully connected.
# 
# <img src="https://i.imgur.com/Hqhg0Q9.png" width="500px">
# 
# 
# ## **Implementation details**
# ```python
# # Regular Convolution, # of params = in_chs * out_chs * kernel_size^2
# nn.Conv2d(in_chs, out_chs, kernel_size, stride, padding)
# 
# # Group Convolution, "groups" controls the connections between inputs and
# # outputs. in_chs and out_chs must both be divisible by groups.
# nn.Conv2d(in_chs, out_chs, kernel_size, stride, padding, groups=groups)
# 
# # Depthwise Convolution, out_chs=in_chs=groups, # of params = in_chs * kernel_size^2
# nn.Conv2d(in_chs, out_chs=in_chs, kernel_size, stride, padding, groups=in_chs)
# 
# # Pointwise Convolution, a.k.a 1 by 1 convolution, # of params = in_chs * out_chs
# nn.Conv2d(in_chs, out_chs, 1)
# 
# # Merge Depthwise and Pointwise Convolution (without )
# def dwpw_conv(in_chs, out_chs, kernel_size, stride, padding):
#     return nn.Sequential(
#         nn.Conv2d(in_chs, in_chs, kernels, stride, padding, groups=in_chs),
#         nn.Conv2d(in_chs, out_chs, 1),
#     )
# ```
# 
# ## **Model**
# 
# The basic model here is simply a stack of convolutional layers followed by some fully-connected layers. You can take advatage of depthwise & pointwise convolution to make your model deeper, but still follow the size constraint.

# In[7]:


class StudentNet(nn.Module):
    def __init__(self):
      super(StudentNet, self).__init__()

      # ---------- TODO ----------
      # Modify your model architecture

      self.cnn = nn.Sequential(
        nn.Conv2d(3, 32, 3), 
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.Conv2d(32, 32, 3),  
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(2, 2, 0),     

        nn.Conv2d(32, 64, 3), 
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(2, 2, 0),     

        nn.Conv2d(64, 100, 3), 
        nn.BatchNorm2d(100),
        nn.ReLU(),
        nn.MaxPool2d(2, 2, 0),
        
        # Here we adopt Global Average Pooling for various input size.
        nn.AdaptiveAvgPool2d((1, 1)),
      )
      self.fc = nn.Sequential(
        nn.Linear(100, 11),
      )
      
    def forward(self, x):
      out = self.cnn(x)
      out = out.view(out.size()[0], -1)
      return self.fc(out)


# In[8]:


#https://github.com/iamhankai/ghostnet.pytorch/blob/master/ghost_net.py
"""
Creates a GhostNet Model as defined in:
GhostNet: More Features from Cheap Operations By Kai Han, Yunhe Wang, Qi Tian, Jianyuan Guo, Chunjing Xu, Chang Xu.
https://arxiv.org/abs/1911.11907
Modified from https://github.com/d-li14/mobilenetv3.pytorch
"""
import torch
import torch.nn as nn
import math


__all__ = ['ghost_net']


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel),        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        y = torch.clamp(y, 0, 1)
        return x * y


def depthwise_conv(inp, oup, kernel_size=3, stride=1, relu=False):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel_size, stride, kernel_size//2, groups=inp, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True) if relu else nn.Sequential(),
    )

class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels*(ratio-1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size//2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size//2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1,x2], dim=1)
        return out[:,:self.oup,:,:]


class GhostBottleneck(nn.Module):
    def __init__(self, inp, hidden_dim, oup, kernel_size, stride, use_se):
        super(GhostBottleneck, self).__init__()
        assert stride in [1, 2]

        self.conv = nn.Sequential(
            # pw
            GhostModule(inp, hidden_dim, kernel_size=1, relu=True),
            # dw
            depthwise_conv(hidden_dim, hidden_dim, kernel_size, stride, relu=False) if stride==2 else nn.Sequential(),
            # Squeeze-and-Excite
            SELayer(hidden_dim) if use_se else nn.Sequential(),
            # pw-linear
            GhostModule(hidden_dim, oup, kernel_size=1, relu=False),
        )

        if stride == 1 and inp == oup:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                depthwise_conv(inp, inp, kernel_size, stride, relu=False),
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)


class GhostNet(nn.Module):
    def __init__(self, cfgs, num_classes=11, width_mult=1.):
        super(GhostNet, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = cfgs

        # building first layer
        output_channel = _make_divisible(16 * width_mult, 4)
        layers = [nn.Sequential(
            nn.Conv2d(3, output_channel, 3, 2, 1, bias=False),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(inplace=True)
        )]
        input_channel = output_channel

        # building inverted residual blocks
        block = GhostBottleneck
        for k, exp_size, c, use_se, s in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 4)
            hidden_channel = _make_divisible(exp_size * width_mult, 4)
            layers.append(block(input_channel, hidden_channel, output_channel, k, s, use_se))
            input_channel = output_channel
        self.features = nn.Sequential(*layers)

        # building last several layers
        output_channel = _make_divisible(exp_size * width_mult, 4)
        self.squeeze = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, 1, 1, 0, bias=False),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        input_channel = output_channel

        output_channel = args.h_dim
        self.classifier = nn.Sequential(
            nn.Linear(input_channel, output_channel, bias=False),
            nn.BatchNorm1d(output_channel),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(output_channel, num_classes),
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.squeeze(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class GhostNet2(nn.Module):
    def __init__(self, cfgs, num_classes=11, width_mult=1.):
        super(GhostNet2, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = cfgs

        # building first layer
        output_channel = _make_divisible(16 * width_mult, 4)
        layers = [nn.Sequential(
            nn.Conv2d(3, output_channel, 3, 2, 1, bias=False),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(inplace=True)
        )]
        input_channel = output_channel

        # building inverted residual blocks
        block = GhostBottleneck
        for k, exp_size, c, use_se, s in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 4)
            hidden_channel = _make_divisible(exp_size * width_mult, 4)
            layers.append(block(input_channel, hidden_channel, output_channel, k, s, use_se))
            input_channel = output_channel
        # for i , l in enumerate(layers):
        #     print(i)
        #     print(l)
        #     print('==========================================')
        # print(len(layers))
        # exit()
        self.features1 = nn.Sequential(*layers[:4])
        self.features2 = nn.Sequential(*layers[4:])

        # building last several layers
        output_channel = _make_divisible(exp_size * width_mult, 4)
        self.squeeze = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, 1, 1, 0, bias=False),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        input_channel = output_channel

        output_channel = args.h_dim
        self.classifier = nn.Sequential(
            nn.Linear(input_channel, output_channel, bias=False),
            nn.BatchNorm1d(output_channel),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(output_channel, num_classes),
        )

        self._initialize_weights()

    def forward(self, x):
        x2 = self.features1(x)
        x = self.features2(x2)
        x = self.squeeze(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x2, x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

def ghost_net(width_multi=1, aux=False, **kwargs):
    """
    Constructs a GhostNet model
    """
    cfgs = [
        # k, t, c, SE, s 
        [3,  16,  16, 0, 1],
        [3,  48,  24, 0, 2],
        [3,  72,  24, 0, 1],
        [5,  72,  40, 1, 2],
        [5, 120,  40, 1, 1],
        [3, 240,  80, 0, 2],
        [3, 200,  80, 0, 1],
#         [3, 184,  80, 0, 1],
#         [3, 184,  80, 0, 1],
#         [3, 480, 112, 1, 1],
#         [3, 672, 112, 1, 1],
#         [5, 672, 160, 1, 2],
#         [5, 960, 160, 0, 1],
#         [5, 960, 160, 1, 1],
#         [5, 960, 160, 0, 1],
#         [5, 960, 160, 1, 1]
    ]
    for i in range(len(cfgs)):
        cfgs[i][1] = int(cfgs[i][1]*width_multi)
        cfgs[i][2] = int(cfgs[i][2]*width_multi)

    if aux == 1:
        return GhostNet2(cfgs, **kwargs)
    elif aux == 0:
        return GhostNet(cfgs, **kwargs)

class AuxNet(nn.Module):
    def __init__(self, input_channel, output_channel, num_classes=11):
        super(AuxNet, self).__init__()

        self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            # nn.AdaptiveAvgPool2d((1, 1)),
            nn.Linear(input_channel, output_channel, bias=False),
            nn.BatchNorm1d(output_channel),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(output_channel, num_classes),
        )

        self._initialize_weights()


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.pooling(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# In[9]:


from torchsummary import summary
student_net = ghost_net(width_multi=1, aux=args.aux_clf)
print(summary(student_net, (3, img_size, img_size), device="cpu"))
start_epoch = 0


aux_net = None
if args.aux_clf == 1:
    aux_net = AuxNet(input_channel=24, output_channel=128, num_classes=11)

if args.resume:
    ckpt_path = os.path.join(args.resume, 'model.ckpt')
    state_dict = torch.load(ckpt_path)
    student_net.load_state_dict(state_dict)


def loss_fn_kd(outputs, labels, teacher_outputs, alpha=args.alpha, T=args.T):
    hard_loss = F.cross_entropy(outputs, labels) * (1. - alpha) 
    # ---------- TODO ----------
    # Complete soft loss in knowledge distillation
    soft_loss = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(outputs/T, dim=1),
                             F.softmax(teacher_outputs/T, dim=1)) * (alpha * T * T)
    
    return hard_loss + soft_loss


# ## **Teacher Model Setting**
# We provide a well-trained teacher model to help you knowledge distillation to student model.
# Note that if you want to change the transform function, you should consider  if suitable for this well-trained teacher model.
# * If you cannot successfully gdown, you can change a link. (Backup link is provided at the bottom of this colab tutorial).
# 

# In[12]:


# Download teacherNet
# get_ipython().system("gdown --id '1zH1x39Y8a0XyOORG7TWzAnFf_YPY8e-m' --output teacher_net.ckpt")
# Load teacherNet

if args.teacher:
    teacher_net = ghost_net(width_multi=1)
    ckpt_path = os.path.join(args.teacher, 'model.ckpt')
    state_dict = torch.load(ckpt_path)
    teacher_net.load_state_dict(state_dict)
else:
    teacher_net = torch.load('./teacher_net.ckpt')
teacher_net.eval()

print(teacher_net)


# ## **Generate Pseudo Labels in Unlabeled Data**
# 
# Since we have a well-trained model, we can use this model to predict pseudo-labels and help the student network train well. Note that you 
# **CANNOT** use well-trained model to pseudo-label the test data. 
# 
# 
# ---
# 
# **AGAIN, DO NOT USE TEST DATA FOR PURPOSE OTHER THAN INFERENCING**
# 
# * Because If you use teacher network to predict pseudo-labels of the test data, you can only use student network to overfit these pseudo-labels without train/unlabeled data. In this way, your kaggle accuracy will be as high as the teacher network, but the fact is that you just overfit the test data and your true testing accuracy is very low. 
# * These contradict the purpose of these assignment (network compression); therefore, you should not misuse the test data.
# * If you have any concerns, you can email us.
# 

# In[13]:


# "cuda" only when GPUs are available.
device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize a model, and put it on the device specified.
student_net = student_net.to(device)
teacher_net = teacher_net.to(device)
if aux_net:
    aux_net = aux_net.to(device)

# Whether to do pseudo label.
do_semi = args.do_semi

def get_pseudo_labels(dataset, model):
    loader = DataLoader(dataset, batch_size=batch_size*3, shuffle=False, pin_memory=True)
    pseudo_labels = []
    for batch in tqdm(loader):
        # A batch consists of image data and corresponding labels.
        img, _ = batch

        # Forward the data
        # Using torch.no_grad() accelerates the forward process.
        with torch.no_grad():
            logits = model(img.to(device))
            pseudo_labels.append(logits.argmax(dim=-1).detach().cpu())
        # Obtain the probability distributions by applying softmax on logits.
    pseudo_labels = torch.cat(pseudo_labels)
    # Update the labels by replacing with pseudo labels.
    for idx, ((img, _), pseudo_label) in enumerate(zip(dataset.samples, pseudo_labels)):
        dataset.samples[idx] = (img, pseudo_label.item())
    return dataset

if do_semi == 1:
    # Generate new trainloader with unlabeled set.
    unlabeled_set = get_pseudo_labels(unlabeled_set, teacher_net)
    concat_dataset = ConcatDataset([train_set, unlabeled_set])
    train_loader = DataLoader(concat_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=True)
elif do_semi == 2:
    # Generate new trainloader with unlabeled set.
    unlabeled_set = get_pseudo_labels(unlabeled_set, teacher_net)
    concat_dataset = ConcatDataset([train_set, unlabeled_set])
    concat_dataset = ConcatDataset([concat_dataset, valid_set])
    train_loader = DataLoader(concat_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=True)

# ## **Training** *(similar to HW3)*
# 
# You can finish supervised learning by simply running the provided code without any modification.
# 
# The function "get_pseudo_labels" is used for semi-supervised learning.
# It is expected to get better performance if you use unlabeled data for semi-supervised learning.
# However, you have to implement the function on your own and need to adjust several hyperparameters manually.
# 
# For more details about semi-supervised learning, please refer to [Prof. Lee's slides](https://speech.ee.ntu.edu.tw/~tlkagk/courses/ML_2016/Lecture/semi%20(v3).pdf).
# 
# Again, please notice that utilizing external data (or pre-trained model) for training is **prohibited**.
# 
# ---
# **The only diffference with HW3 is that you should use loss in  knowledge distillation.**
# 
# 
# 

# In[14]:


# For the classification task, we use cross-entropy as the measurement of performance.
criterion = nn.CrossEntropyLoss()

# Initialize optimizer, you may fine-tune some hyperparameters such as learning rate on your own.
# optimizer = torch.optim.Adam(student_net.parameters(), lr=lr, weight_decay=1e-5)

from timm.optim.optim_factory import create_optimizer
optimizer = create_optimizer(args, student_net)
if args.aux_clf == 1:
    optimizer2 = create_optimizer(args, aux_net)

if args.opt_scheduler == 'cos':
    from timm.scheduler.cosine_lr import CosineLRScheduler
    scheduler = CosineLRScheduler(optimizer, t_initial=args.opt_scheduler_tinit, cycle_limit=args.opt_scheduler_cycle,\
                                     warmup_t=args.opt_scheduler_warmup_t, decay_rate=1., t_mul=1, lr_min=args.opt_scheduler_minlr, warmup_lr_init=1e-5)
    if args.aux_clf == 1:
        scheduler2 = CosineLRScheduler(optimizer2, t_initial=args.opt_scheduler_tinit, cycle_limit=args.opt_scheduler_cycle,\
                                         warmup_t=args.opt_scheduler_warmup_t, decay_rate=1., t_mul=1, lr_min=args.opt_scheduler_minlr, warmup_lr_init=1e-5)
    def get_lr_per_epoch(scheduler, num_epoch):
        lr_per_epoch = []
        for epoch in range(num_epoch):
            lr_per_epoch.append(scheduler.get_epoch_values(epoch))
        return lr_per_epoch
    lr_per_epoch = get_lr_per_epoch(scheduler, args.epochs)
    import matplotlib.pyplot as plt
    plt.plot(lr_per_epoch)
    plt.savefig(os.path.join(log_path, 'lr.png'))
else:
    scheduler = None
    scheduler2 = None
# The number of training epochs.
n_epochs = args.epochs



model_path = os.path.join(log_path, './model.ckpt')
model_path2 = os.path.join(log_path, './model_last.ckpt')
best_acc = 0.0

if args.aux_clf == 1:

    train_history = {'epoch':[], 'loss':[], 'accuracy':[], 'accuracy2':[]}
    valid_history = {'epoch':[], 'loss':[], 'accuracy':[], 'accuracy2':[]}
    for epoch in range(start_epoch, n_epochs):
        # ---------- Training ----------
        # Make sure the model is in train mode before training.
        student_net.train()
        aux_net.train()

        # These are used to record information in training.
        train_loss = []
        train_accs = []
        train_accs2 = []

        # Iterate the training set by batches.
        for batch in tqdm(train_loader):

            # A batch consists of image data and corresponding labels.
            imgs, labels = batch

            # Forward the data. (Make sure data and model are on the same device.)
            logits1, logits2 = student_net(imgs.to(device))
            logits1 = aux_net(logits1)
            # Teacher net will not be updated. And we use torch.no_grad
            # to tell torch do not retain the intermediate values
            # (which are for backpropgation) and save the memory.
            with torch.no_grad():
              soft_labels = teacher_net(imgs.to(device))
            
            # Calculate the loss in knowledge distillation method.
            loss = loss_fn_kd(logits1, labels.to(device), soft_labels)
            loss2 = loss_fn_kd(logits2, labels.to(device), soft_labels)

            loss = loss + loss2
            # Gradients stored in the parameters in the previous step should be cleared out first.
            optimizer.zero_grad()
            optimizer2.zero_grad()

            # Compute the gradients for parameters.
            loss.backward()

            # Clip the gradient norms for stable training.
            grad_norm = nn.utils.clip_grad_norm_(student_net.parameters(), max_norm=10)

            # Update the parameters with computed gradients.
            optimizer.step()
            optimizer2.step()

            # Compute the accuracy for current batch.
            acc1 = (logits1.argmax(dim=-1) == labels.to(device)).float().mean()
            acc2 = (logits2.argmax(dim=-1) == labels.to(device)).float().mean()

            # Record the loss and accuracy.
            train_loss.append(loss.item())
            train_accs.append(acc1)
            train_accs2.append(acc2)

        # The average loss and accuracy of the training set is the average of the recorded values.
        train_loss = sum(train_loss) / len(train_loss)
        train_acc = sum(train_accs) / len(train_accs)
        train_acc2 = sum(train_accs2) / len(train_accs2)

        # Print the information.
        print(f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}, acc2 = {train_acc2:.5f}")
        train_history['epoch'].append(epoch)
        train_history['loss'].append(train_loss)
        train_history['accuracy'].append(train_acc)
        train_history['accuracy2'].append(train_acc2)

        # ---------- Validation ----------
        # Make sure the model is in eval mode so that some modules like dropout are disabled and work normally.
        student_net.eval()

        # These are used to record information in validation.
        valid_loss = []
        valid_accs = []
        valid_accs2 = []

        # Iterate the validation set by batches.
        for batch in tqdm(valid_loader):

            # A batch consists of image data and corresponding labels.
            imgs, labels = batch

            # We don't need gradient in validation.
            # Using torch.no_grad() accelerates the forward process.
            with torch.no_grad():
              logits1, logits2 = student_net(imgs.to(device))
              logits1 = aux_net(logits1)

              soft_labels = teacher_net(imgs.to(device))
            # We can still compute the loss (but not the gradient).
            loss = loss_fn_kd(logits1, labels.to(device), soft_labels)
            loss2 = loss_fn_kd(logits2, labels.to(device), soft_labels)

            # Compute the accuracy for current batch.
            acc = (logits1.argmax(dim=-1) == labels.to(device)).float().detach().cpu().view(-1).numpy()
            acc2 = (logits2.argmax(dim=-1) == labels.to(device)).float().detach().cpu().view(-1).numpy()

            # Record the loss and accuracy.
            valid_loss.append(loss.item())
            valid_accs += list(acc)
            valid_accs2 += list(acc2)

        # The average loss and accuracy for entire validation set is the average of the recorded values.
        valid_loss = sum(valid_loss) / len(valid_loss)
        valid_acc = sum(valid_accs) / len(valid_accs)
        valid_acc2 = sum(valid_accs2) / len(valid_accs2)

        valid_history['epoch'].append(epoch)
        valid_history['loss'].append(valid_loss)
        valid_history['accuracy'].append(valid_acc)
        valid_history['accuracy2'].append(valid_acc2)
        
        if valid_acc2 > best_acc:
            best_acc = valid_acc2
            torch.save(student_net.state_dict(), model_path)
            print('saving model with acc {:.3f}'.format(best_acc))
        if epoch ==  n_epochs-1:
            torch.save(student_net.state_dict(), model_path2)
            print('saving last model with acc {:.3f}'.format(valid_acc2))

        if scheduler is not None:
            scheduler.step(epoch)
        if scheduler2 is not None:
            scheduler2.step(epoch)
        # Print the information.
        print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}, acc2 = {valid_acc2:.5f}")

        train_history_df = pd.DataFrame(train_history)
        valid_history_df = pd.DataFrame(valid_history)
        import plotly.express as px
        fig = px.line(train_history_df, x="epoch", y="accuracy2", title='train accuracy')
        fig.write_image(os.path.join(log_path, 'train_history.png'))

        fig = px.line(valid_history_df, x="epoch", y="accuracy2", title='valid accuracy')
        fig.write_image(os.path.join(log_path, 'valid_history.png'))

        train_history_df.to_csv(os.path.join(log_path, 'train_history.csv'), index=False)
        valid_history_df.to_csv(os.path.join(log_path, 'valid_history.csv'), index=False)
elif args.aux_clf == 0:

    train_history = {'epoch':[], 'loss':[], 'accuracy':[]}
    valid_history = {'epoch':[], 'loss':[], 'accuracy':[]}
    for epoch in range(start_epoch, n_epochs):
        # ---------- Training ----------
        # Make sure the model is in train mode before training.
        student_net.train()

        # These are used to record information in training.
        train_loss = []
        train_accs = []

        # Iterate the training set by batches.
        for batch in tqdm(train_loader):

            # A batch consists of image data and corresponding labels.
            imgs, labels = batch

            # Forward the data. (Make sure data and model are on the same device.)
            logits = student_net(imgs.to(device))
            # Teacher net will not be updated. And we use torch.no_grad
            # to tell torch do not retain the intermediate values
            # (which are for backpropgation) and save the memory.
            with torch.no_grad():
              soft_labels = teacher_net(imgs.to(device))
            
            # Calculate the loss in knowledge distillation method.
            loss = loss_fn_kd(logits, labels.to(device), soft_labels)

            # Gradients stored in the parameters in the previous step should be cleared out first.
            optimizer.zero_grad()

            # Compute the gradients for parameters.
            loss.backward()

            # Clip the gradient norms for stable training.
            grad_norm = nn.utils.clip_grad_norm_(student_net.parameters(), max_norm=10)

            # Update the parameters with computed gradients.
            optimizer.step()

            # Compute the accuracy for current batch.
            acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

            # Record the loss and accuracy.
            train_loss.append(loss.item())
            train_accs.append(acc)

        # The average loss and accuracy of the training set is the average of the recorded values.
        train_loss = sum(train_loss) / len(train_loss)
        train_acc = sum(train_accs) / len(train_accs)

        # Print the information.
        print(f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")
        train_history['epoch'].append(epoch)
        train_history['loss'].append(train_loss)
        train_history['accuracy'].append(train_acc)

        # ---------- Validation ----------
        # Make sure the model is in eval mode so that some modules like dropout are disabled and work normally.
        student_net.eval()

        # These are used to record information in validation.
        valid_loss = []
        valid_accs = []

        # Iterate the validation set by batches.
        for batch in tqdm(valid_loader):

            # A batch consists of image data and corresponding labels.
            imgs, labels = batch

            # We don't need gradient in validation.
            # Using torch.no_grad() accelerates the forward process.
            with torch.no_grad():
              logits = student_net(imgs.to(device))
              soft_labels = teacher_net(imgs.to(device))
            # We can still compute the loss (but not the gradient).
            loss = loss_fn_kd(logits, labels.to(device), soft_labels)

            # Compute the accuracy for current batch.
            acc = (logits.argmax(dim=-1) == labels.to(device)).float().detach().cpu().view(-1).numpy()

            # Record the loss and accuracy.
            valid_loss.append(loss.item())
            valid_accs += list(acc)

        # The average loss and accuracy for entire validation set is the average of the recorded values.
        valid_loss = sum(valid_loss) / len(valid_loss)
        valid_acc = sum(valid_accs) / len(valid_accs)

        valid_history['epoch'].append(epoch)
        valid_history['loss'].append(valid_loss)
        valid_history['accuracy'].append(valid_acc)
        
        if valid_acc > best_acc:
            best_acc = valid_acc
            torch.save(student_net.state_dict(), model_path)
            print('saving model with acc {:.3f}'.format(best_acc))
        if epoch ==  n_epochs-1:
            torch.save(student_net.state_dict(), model_path2)
            print('saving last model with acc {:.3f}'.format(valid_acc))

        if scheduler is not None:
            scheduler.step(epoch)
        # Print the information.
        print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")

        train_history_df = pd.DataFrame(train_history)
        valid_history_df = pd.DataFrame(valid_history)
        import plotly.express as px
        fig = px.line(train_history_df, x="epoch", y="accuracy", title='train accuracy')
        fig.write_image(os.path.join(log_path, 'train_history.png'))

        fig = px.line(valid_history_df, x="epoch", y="accuracy", title='valid accuracy')
        fig.write_image(os.path.join(log_path, 'valid_history.png'))

        train_history_df.to_csv(os.path.join(log_path, 'train_history.csv'), index=False)
        valid_history_df.to_csv(os.path.join(log_path, 'valid_history.csv'), index=False)




student_net.eval()
# In[15]:
predictions = []

# Iterate the testing set by batches.
for batch in tqdm(test_loader):
    # A batch consists of image data and corresponding labels.
    # But here the variable "labels" is useless since we do not have the ground-truth.
    # If printing out the labels, you will find that it is always 0.
    # This is because the wrapper (DatasetFolder) returns images and labels for each batch,
    # so we have to create fake labels to make it work normally.
    imgs, labels = batch

    # We don't need gradient in testing, and we don't even have labels to compute loss.
    # Using torch.no_grad() accelerates the forward process.
    if args.aux_clf == 1:
        with torch.no_grad():
            _, logits = student_net(imgs.to(device))

    else:
        with torch.no_grad():
            logits = student_net(imgs.to(device))

    # Take the class with greatest logit as prediction and record it.
    predictions.extend(logits.argmax(dim=-1).cpu().numpy().tolist())


# In[16]:


### This block is same as HW3 ###
# Save predictions into the file.
with open(os.path.join(log_path, "predict_last.csv"), "w") as f:

    # The first row must be "Id, Category"
    f.write("Id,Category\n")

    # For the rest of the rows, each image id corresponds to a predicted class.
    for i, pred in  enumerate(predictions):
         f.write(f"{i},{pred}\n")


### This block is same as HW3 ###
# Make sure the model is in eval mode.
# Some modules like Dropout or BatchNorm affect if the model is in training mode.
student_net.load_state_dict(torch.load(os.path.join(log_path, 'model.ckpt')))

student_net.eval()

# Initialize a list to store the predictions.
predictions = []

# Iterate the testing set by batches.
for batch in tqdm(test_loader):
    # A batch consists of image data and corresponding labels.
    # But here the variable "labels" is useless since we do not have the ground-truth.
    # If printing out the labels, you will find that it is always 0.
    # This is because the wrapper (DatasetFolder) returns images and labels for each batch,
    # so we have to create fake labels to make it work normally.
    imgs, labels = batch

    if args.aux_clf == 1:
        with torch.no_grad():
            _, logits = student_net(imgs.to(device))

    else:
        with torch.no_grad():
            logits = student_net(imgs.to(device))
    # Take the class with greatest logit as prediction and record it.
    predictions.extend(logits.argmax(dim=-1).cpu().numpy().tolist())


# In[16]:


### This block is same as HW3 ###
# Save predictions into the file.
with open(os.path.join(log_path, "predict.csv"), "w") as f:

    # The first row must be "Id, Category"
    f.write("Id,Category\n")

    # For the rest of the rows, each image id corresponds to a predicted class.
    for i, pred in  enumerate(predictions):
         f.write(f"{i},{pred}\n")


# ## **Statistics**
# 
# |Baseline|Accuracy|Training Time|
# |-|-|-|
# |Simple Baseline |0.59856|2 Hours|
# |Medium Baseline |0.65412|2 Hours|
# |Strong Baseline |0.72819|4 Hours|
# |Boss Baseline |0.81003|Unmeasueable|
# 
# ## **Learning Curve**
# 
# ![img](https://lh5.googleusercontent.com/amMLGa7dkqvXGmsJlrVN49VfSjClk5d-n7nCi_Y3ROK4himsBSHhB7SpdWe7Zm06ctRO77VdDkD9u_aKfAh1tMW-KcyYX7vF7LPlKqOo2fVtt3SyfsLv0KTYDB0YbAk6ZhyOIKT8Zfg)

# 
# 
# ## **Q&A**
# 
# If you have any question about this colab, please send a email to ntu-ml-2021spring-ta@googlegroups.com

# ## **Backup Links**

# In[17]:


# resnet_model 
# !gdown --id '1zH1x39Y8a0XyOORG7TWzAnFf_YPY8e-m' --output resnet_model.ckpt
# !gdown --id '1VBIeQKH4xRHfToUxuDxtEPsqz0MHvrgd' --output resnet_model.ckpt
# !gdown --id '1Er2azErvXWS5m1jboKN7BLxNXnuAatYw' --output resnet_model.ckpt
# !gdown --id '1Qya0vmf3nRl11IyxxF7nudDpZI_Q4Amh' --output resnet_model.ckpt
# !gdown --id '1fGOOb5ndljraBIkRkLp3bW9orR4YN97U' --output resnet_model.ckpt
# !gdown --id '1apHLvZBZ3GYEMxXxToGKF7qDLn1XbOfJ' --output resnet_model.ckpt
# !gdown --id '1vsDylNsLaAqxonop7Mw3dBAig0EO7tlF' --output resnet_model.ckpt
# !gdown --id '1V_hXJM_V9-10i6wldRyl0SOiivPp4SNt' --output resnet_model.ckpt
# !gdown --id '11HzaJM2M2yg6KYhLaWpWy8WmPIIvJgnk' --output resnet_model.ckpt

# food-11
# !gdown --id '1qdyNN0Ek4S5yi-pAqHes1yjj5cNkENCc' --output food-11.zip
# !gdown --id '1c0Q1EP6yIx0O2rqVMIVInIt8wFjLxmRh' --output food-11.zip
# !gdown --id '1hKO054nT1R8egcXY2-tgQbwX4EjowRLz' --output food-11.zip
# !gdown --id '1_7_uC1WUvX6H51gQaYmI4q3AezdQJhud' --output food-11.zip
# !gdown --id '12bz82Zpx0_7BDGXq4nRt7E_fMFmILoc9' --output food-11.zip
# !gdown --id '1oiqRKrDQXVBM5y63MeEaHxFmCIzNXx1Q' --output food-11.zip
# !gdown --id '1qaL43sl4qUMeCT1OVpk4aOFycnLL5ZJX' --output food-11.zip


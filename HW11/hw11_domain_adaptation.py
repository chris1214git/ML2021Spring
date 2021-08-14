#!/usr/bin/env python
# coding: utf-8

# # Homework 11 - Transfer Learning (Domain Adversarial Training)
# 
# > Author: Arvin Liu (r09922071@ntu.edu.tw)
# 
# 若有任何問題，歡迎來信至助教信箱 ntu-ml-2021spring-ta@googlegroups.com

# # Readme
# 
# 
# 這份作業的任務是Transfer Learning中的Domain Adversarial Training。
# 
# <img src="https://i.imgur.com/iMVIxCH.png" width="500px">
# 
# > 也就是左下角的那一塊。
# 
# ## Scenario and Why Domain Adversarial Training
# 你現在有Source Data + label，其中Source Data和Target Data可能有點關係，所以你想要訓練一個model做在Source Data上並Predict在Target Data上。
# 
# 但這樣有什麼樣的問題? 相信大家學過Anomaly Detection就會知道，如果有data是在Source Data沒有出現過的(或稱Abnormal的)，那麼model大部分都會因為不熟悉這個data而可能亂做一發。 
# 
# 以下我們將model拆成Feature Extractor(上半部)和Classifier(下半部)來作例子:
# <img src="https://i.imgur.com/IL0PxCY.png" width="500px">
# 
# 整個Model在學習Source Data的時候，Feature Extrator因為看過很多次Source Data，所以所抽取出來的Feature可能就頗具意義，例如像圖上的藍色Distribution，已經將圖片分成各個Cluster，所以這個時候Classifier就可以依照這個Cluster去預測結果。
# 
# 但是在做Target Data的時候，Feature Extractor會沒看過這樣的Data，導致輸出的Target Feature可能不屬於在Source Feature Distribution上，這樣的Feature給Classifier預測結果顯然就不會做得好。
# 
# ## Domain Adversarial Training of Nerural Networks (DaNN)
# 基於如此，是不是只要讓Soucre Data和Target Data經過Feature Extractor都在同個Distribution上，就會做得好了呢? 這就是DaNN的主要核心。
# 
# <img src="https://i.imgur.com/vrOE5a6.png" width="500px">
# 
# 我們追加一個Domain Classifier，在學習的過程中，讓Domain Classifier去判斷經過Feature Extractor後的Feature是源自於哪個domain，讓Feature Extractor學習如何產生Feature以**騙過**Domain Classifier。 持久下來，通常Feature Extractor都會打贏Domain Classifier。(因為Domain Classifier的Input來自於Feature Extractor，而且對Feature Extractor來說Domain&Classification的任務並沒有衝突。)
# 
# 如此一來，我們就可以確信不管是哪一個Domain，Feature Extractor都會把它產生在同一個Feature Distribution上。

# # Data Introduce
# 
# 這次的任務是Source Data: 真實照片，Target Data: 手畫塗鴉。
# 
# 我們必須讓model看過真實照片以及標籤，嘗試去預測手畫塗鴉的標籤為何。
# 
# 資料位於[這裡](https://drive.google.com/open?id=12-07DSquGdzN3JBHBChN4nMo3i8BqTiL)，以下的code分別為下載和觀看這次的資料大概長甚麼樣子。
# 
# 特別注意一點: **這次的source和target data的圖片都是平衡的，你們可以使用這個資訊做其他事情。**

# In[1]:


# # Download dataset
# !gdown --id '1P4fGNb9JhJj8W0DA_Qrp7mbrRHfF5U_f' --output real_or_drawing.zip
# # Unzip the files
# !unzip real_or_drawing.zip


# In[2]:

import cv2


# # Data Process
# 
# 在這裡我故意將data用成可以使用torchvision.ImageFolder的形式，所以只要使用該函式便可以做出一個datasets。
# 
# transform的部分請參考以下註解。
# <!-- 
# #### 一些細節
# 
# 在一般的版本上，對灰階圖片使用RandomRotation使用```transforms.RandomRotation(15)```即可。但在colab上需要加上```fill=(0,)```才可運行。
# 在n98上執行需要把```fill=(0,)```拿掉才可運行。 -->
# 

# In[5]:

import os
import argparse
import sys
import json

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='DaNN')
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--batch_size', type=int, default=32)

# parser.add_argument('--augment', type=str, default=None)
parser.add_argument('--augment', type=int, default=0)

parser.add_argument('--Canny_low', type=int, default=170)
parser.add_argument('--Canny_high', type=int, default=300)
parser.add_argument('--lamb_decay', type=float, default=10)

parser.add_argument('--opt', type=str, default='adam', choices=['sgd', 'adam', 'adamw', 'adamp'])
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--weight_decay', type=float, default=0)
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
parser.add_argument('--do_semi', type=int, default=0)
parser.add_argument('--semi_batch_size', type=int, default=32)
parser.add_argument('--semi_epoch', type=int, default=50)
parser.add_argument('--semi_init_epoch', type=float, default=300)
parser.add_argument('--threshold', type=float, default=0.5)
parser.add_argument('--select_num', type=int, default=1000)

parser.add_argument('--cuda', type=int, default=0)
parser.add_argument('--log_dir', type=str, default='DaNN-1')
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
# In[6]:


Canny_setting = [args.Canny_low, args.Canny_high]
# In[7]:


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

if args.augment == 0:
    source_transform = transforms.Compose([
        # Turn RGB to grayscale. (Bacause Canny do not support RGB images.)
        transforms.Grayscale(),
        # cv2 do not support skimage.Image, so we transform it to np.array, 
        # and then adopt cv2.Canny algorithm.
        transforms.Lambda(lambda x: cv2.Canny(np.array(x), Canny_setting[0], Canny_setting[1])),
        # Transform np.array back to the skimage.Image.
        transforms.ToPILImage(),
        # 50% Horizontal Flip. (For Augmentation)
        
    #     transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        # Rotate +- 15 degrees. (For Augmentation), and filled with zero 
        # if there's empty pixel after rotation.
        transforms.RandomRotation(15, fill=(0,)),

        # Transform to tensor for model inputs.
        transforms.ToTensor(),
    ])
    target_transform = transforms.Compose([
        # Turn RGB to grayscale.
        transforms.Grayscale(),
        # Resize: size of source data is 32x32, thus we need to 
        #  enlarge the size of target data from 28x28 to 32x32。
        transforms.Resize((32, 32)),
        # 50% Horizontal Flip. (For Augmentation)
        
        transforms.RandomHorizontalFlip(),
        # Rotate +- 15 degrees. (For Augmentation), and filled with zero 
        # if there's empty pixel after rotation.
        transforms.RandomRotation(15, fill=(0,)),
        # Transform to tensor for model inputs.
        transforms.ToTensor(),
    ])
elif args.augment == 1:
    source_transform = transforms.Compose([
        # Turn RGB to grayscale. (Bacause Canny do not support RGB images.)
        transforms.Grayscale(),
        # cv2 do not support skimage.Image, so we transform it to np.array, 
        # and then adopt cv2.Canny algorithm.
        transforms.Lambda(lambda x: cv2.Canny(np.array(x), Canny_setting[0], Canny_setting[1])),
        # Transform np.array back to the skimage.Image.
        transforms.ToPILImage(),
        # 50% Horizontal Flip. (For Augmentation)
        
    #     transforms.RandomCrop(32, padding=4),
        transforms.RandomAffine(10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(),
        # Rotate +- 15 degrees. (For Augmentation), and filled with zero 
        # if there's empty pixel after rotation.
        # transforms.RandomRotation(15, fill=(0,)),

        # Transform to tensor for model inputs.
        transforms.ToTensor(),
    ])
    target_transform = transforms.Compose([
        # Turn RGB to grayscale.
        transforms.Grayscale(),
        # Resize: size of source data is 32x32, thus we need to 
        #  enlarge the size of target data from 28x28 to 32x32。
        transforms.Resize((32, 32)),
        # 50% Horizontal Flip. (For Augmentation)
        transforms.RandomAffine(10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(),
        # Rotate +- 15 degrees. (For Augmentation), and filled with zero 
        # if there's empty pixel after rotation.
        # transforms.RandomRotation(15, fill=(0,)),
        # Transform to tensor for model inputs.
        transforms.ToTensor(),
    ])
        
test_transform = transforms.Compose([
    # Turn RGB to grayscale.
    transforms.Grayscale(),
    # Resize: size of source data is 32x32, thus we need to 
    #  enlarge the size of target data from 28x28 to 32x32。
    transforms.Resize((32, 32)),
    # 50% Horizontal Flip. (For Augmentation)
    
    # Transform to tensor for model inputs.
    transforms.ToTensor(),
])

source_dataset = ImageFolder('real_or_drawing/train_data', transform=source_transform)
target_dataset = ImageFolder('real_or_drawing/test_data', transform=target_transform)
if args.modified_test:
    test_dataset = ImageFolder('real_or_drawing/test_data', transform=test_transform)
else:
    test_dataset = ImageFolder('real_or_drawing/test_data', transform=target_transform)

print(test_dataset)
source_dataloader = DataLoader(source_dataset, batch_size=32, shuffle=True, num_workers=1)
target_dataloader = DataLoader(target_dataset, batch_size=32, shuffle=True, num_workers=1)
test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=1)

target_semi_dataloader = None
# # Model
# 
# Feature Extractor: 典型的VGG-like疊法。
# 
# Label Predictor / Domain Classifier: MLP到尾。
# 
# 相信作業寫到這邊大家對以下的Layer都很熟悉，因此不再贅述。

# In[8]:


import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, is_last=False):
        super(BasicBlock, self).__init__()
        self.is_last = is_last
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        preact = out
        out = F.relu(out)
        if self.is_last:
            return out, preact
        else:
            return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, is_last=False):
        super(Bottleneck, self).__init__()
        self.is_last = is_last
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        preact = out
        out = F.relu(out)
        if self.is_last:
            return out, preact
        else:
            return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def get_feat_modules(self):
        feat_m = nn.ModuleList([])
        feat_m.append(self.conv1)
        feat_m.append(self.bn1)
        feat_m.append(self.layer1)
        feat_m.append(self.layer2)
        feat_m.append(self.layer3)
        feat_m.append(self.layer4)
        return feat_m

    def get_bn_before_relu(self):
        if isinstance(self.layer1[0], Bottleneck):
            bn1 = self.layer1[-1].bn3
            bn2 = self.layer2[-1].bn3
            bn3 = self.layer3[-1].bn3
            bn4 = self.layer4[-1].bn3
        elif isinstance(self.layer1[0], BasicBlock):
            bn1 = self.layer1[-1].bn2
            bn2 = self.layer2[-1].bn2
            bn3 = self.layer3[-1].bn2
            bn4 = self.layer4[-1].bn2
        else:
            raise NotImplementedError('ResNet unknown block error !!!')

        return [bn1, bn2, bn3, bn4]

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for i in range(num_blocks):
            stride = strides[i]
            layers.append(block(self.in_planes, planes, stride, i == num_blocks - 1))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, is_feat=False, preact=False):
        out = F.relu(self.bn1(self.conv1(x)))
        f0 = out
        out, f1_pre = self.layer1(out)
        f1 = out
        out, f2_pre = self.layer2(out)
        f2 = out
        out, f3_pre = self.layer3(out)
        f3 = out
        out, f4_pre = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        return out

def ResNet18(**kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)

class FeatureExtractor(nn.Module):

    def __init__(self):
        super(FeatureExtractor, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
    def forward(self, x):
        x = self.conv(x).squeeze()
        return x

class LabelPredictor(nn.Module):

    def __init__(self):
        super(LabelPredictor, self).__init__()

        self.layer = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.ReLU(),

            nn.Linear(512, 10),
        )

    def forward(self, h):
        c = self.layer(h)
        return c

class DomainClassifier(nn.Module):

    def __init__(self):
        super(DomainClassifier, self).__init__()

        self.layer = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 1),
        )

    def forward(self, h):
        y = self.layer(h)
        return y


# # Pre-processing
# 
# 這裡我們選用Adam來當Optimizer。

# In[9]:


start_epoch = 0
if args.model == 'DaNN':
    feature_extractor = FeatureExtractor().cuda()
elif args.model == 'resnet18':
    feature_extractor = ResNet18().cuda()

label_predictor = LabelPredictor().cuda()
domain_classifier = DomainClassifier().cuda()

if args.resume:

    ckpt_path = os.path.join(args.resume, 'feature_extractor.bin')
    state_dict = torch.load(ckpt_path)
    feature_extractor.load_state_dict(state_dict)

    ckpt_path = os.path.join(args.resume, 'label_predictor.bin')
    state_dict = torch.load(ckpt_path)
    label_predictor.load_state_dict(state_dict)

    ckpt_path = os.path.join(args.resume, 'domain_classifier.bin')
    state_dict = torch.load(ckpt_path)
    domain_classifier.load_state_dict(state_dict)

    import json
    with open(os.path.join(args.resume, 'para.txt'), 'r') as f:
        cmd = json.load(f)
        start_epoch = int(cmd['epochs'])
    
class LabelSmoothing(nn.Module):
    """NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.0):
        """Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

if args.crit == 'ce':
    class_criterion = nn.CrossEntropyLoss()
elif args.crit == 'ls':
    class_criterion = LabelSmoothing(smoothing=args.ls_smooth)
domain_criterion = nn.BCEWithLogitsLoss()

    

if args.origin_opt:
    optimizer_F = optim.Adam(feature_extractor.parameters())
    optimizer_C = optim.Adam(label_predictor.parameters())
    optimizer_D = optim.Adam(domain_classifier.parameters())
else:
    from timm.optim.optim_factory import create_optimizer
    optimizer_F = create_optimizer(args, feature_extractor)
    optimizer_C = create_optimizer(args, label_predictor)
    optimizer_D = create_optimizer(args, domain_classifier)

# if args.opt_scheduler == 'cos':
#     from timm.scheduler.cosine_lr import CosineLRScheduler
#     scheduler = CosineLRScheduler(optimizer, t_initial=args.opt_scheduler_tinit, cycle_limit=args.opt_scheduler_cycle,\
#                                      warmup_t=args.opt_scheduler_warmup_t, decay_rate=1., t_mul=1, lr_min=args.opt_scheduler_minlr, warmup_lr_init=1e-5)
#     def get_lr_per_epoch(scheduler, num_epoch):
#         lr_per_epoch = []
#         for epoch in range(num_epoch):
#             lr_per_epoch.append(scheduler.get_epoch_values(epoch))
#         return lr_per_epoch
#     lr_per_epoch = get_lr_per_epoch(scheduler, 50*2)
#     import matplotlib.pyplot as plt
#     plt.plot(lr_per_epoch)
#     plt.savefig(os.path.join(log_path, 'lr.png'))
# else:
#     scheduler = None

# # Start Training
# 
# 
# ## 如何實作DaNN?
# 
# 理論上，在原始paper中是加上Gradient Reversal Layer，並將Feature Extractor / Label Predictor / Domain Classifier 一起train，但其實我們也可以交換的train Domain Classfier & Feature Extractor(就像在train GAN的Generator & Discriminator一樣)，這也是可行的。
# 
# 在code實現中，我們採取後者的方式，畢竟GAN是之前的作業，應該會比較熟悉:)。
# 
# ## 小提醒
# * 原文中的lambda(控制Domain Adversarial Loss的係數)是有Adaptive的版本，如果有興趣可以參考[原文](https://arxiv.org/pdf/1505.07818.pdf)。
# * 因為我們完全沒有target的label，所以結果如何，只好丟kaggle看看囉:)?

# In[10]:


import math
total_epoch = args.epochs
lamb_decay = args.lamb_decay


# In[ ]:
def train_epoch(source_dataloader, target_dataloader, lamb):
    '''
      Args:
        source_dataloader: source data的dataloader
        target_dataloader: target data的dataloader
        lamb: control the balance of domain adaptatoin and classification.
    '''

    # D loss: Domain Classifier的loss
    # F loss: Feature Extrator & Label Predictor的loss
    running_D_loss, running_F_loss = 0.0, 0.0
    total_hit, total_num = 0.0, 0.0

    for i, ((source_data, source_label), (target_data, _)) in enumerate(zip(source_dataloader, target_dataloader)):

        source_data = source_data.cuda()
        source_label = source_label.cuda()
        target_data = target_data.cuda()
        
        # Mixed the source data and target data, or it'll mislead the running params
        #   of batch_norm. (runnning mean/var of soucre and target data are different.)
        mixed_data = torch.cat([source_data, target_data], dim=0)
        domain_label = torch.zeros([source_data.shape[0] + target_data.shape[0], 1]).cuda()
        # set domain label of source data to be 1.
        domain_label[:source_data.shape[0]] = 1

        # Step 1 : train domain classifier
        feature = feature_extractor(mixed_data)
        # We don't need to train feature extractor in step 1.
        # Thus we detach the feature neuron to avoid backpropgation.
        # print(feature.shape)

        domain_logits = domain_classifier(feature.detach())
        loss = domain_criterion(domain_logits, domain_label)
        running_D_loss+= loss.item()
        loss.backward()
        optimizer_D.step()

        # Step 2 : train feature extractor and label classifier
        class_logits = label_predictor(feature[:source_data.shape[0]])
        domain_logits = domain_classifier(feature)
        # loss = cross entropy of classification - lamb * domain binary cross entropy.
        #  The reason why using subtraction is similar to generator loss in disciminator of GAN
        loss = class_criterion(class_logits, source_label) - lamb * domain_criterion(domain_logits, domain_label)
        running_F_loss+= loss.item()
        loss.backward()
        optimizer_F.step()
        optimizer_C.step()

        optimizer_D.zero_grad()
        optimizer_F.zero_grad()
        optimizer_C.zero_grad()

        total_hit += torch.sum(torch.argmax(class_logits, dim=1) == source_label).item()
        total_num += source_data.shape[0]
        print(i, end='\r')

    return running_D_loss / (i+1), running_F_loss / (i+1), total_hit / total_num


def train_semi_epoch(source_dataloader, target_dataloader, target_semi_dataloader, lamb):
    '''
      Args:
        source_dataloader: source data的dataloader
        target_dataloader: target data的dataloader
        lamb: control the balance of domain adaptatoin and classification.
    '''

    # D loss: Domain Classifier的loss
    # F loss: Feature Extrator & Label Predictor的loss
    running_D_loss, running_F_loss = 0.0, 0.0
    total_hit, total_num = 0.0, 0.0

    for i, ((source_data, source_label), (target_data, _), (target_semi_data, target_semi_label)) in enumerate(zip(source_dataloader, target_dataloader, target_semi_dataloader)):

        source_data = source_data.cuda()
        source_label = source_label.cuda()
        target_data = target_data.cuda()
        
        target_semi_data = target_semi_data.cuda()
        target_semi_label = target_semi_label.cuda()

        # Mixed the source data and target data, or it'll mislead the running params
        #   of batch_norm. (runnning mean/var of soucre and target data are different.)
        mixed_data = torch.cat([source_data, target_data], dim=0)
        domain_label = torch.zeros([source_data.shape[0] + target_data.shape[0], 1]).cuda()
        # set domain label of source data to be 1.
        domain_label[:source_data.shape[0]] = 1

        # Step 1 : train domain classifier
        feature = feature_extractor(mixed_data)
        # We don't need to train feature extractor in step 1.
        # Thus we detach the feature neuron to avoid backpropgation.
        # print(feature.shape)

        domain_logits = domain_classifier(feature.detach())
        loss = domain_criterion(domain_logits, domain_label)
        running_D_loss+= loss.item()
        loss.backward()
        optimizer_D.step()

        # Step 2 : train feature extractor and label classifier
        class_logits = label_predictor(feature[:source_data.shape[0]])
        domain_logits = domain_classifier(feature)
        # loss = cross entropy of classification - lamb * domain binary cross entropy.
        #  The reason why using subtraction is similar to generator loss in disciminator of GAN
        loss = class_criterion(class_logits, source_label) - lamb * domain_criterion(domain_logits, domain_label)
        
        feature_semi = feature_extractor(target_semi_data)
        class_logits_semi = label_predictor(feature_semi)
        loss += class_criterion(class_logits_semi, target_semi_label)

        running_F_loss+= loss.item()
        loss.backward()
        optimizer_F.step()
        optimizer_C.step()

        optimizer_D.zero_grad()
        optimizer_F.zero_grad()
        optimizer_C.zero_grad()

        total_hit += torch.sum(torch.argmax(class_logits, dim=1) == source_label).item()
        total_num += source_data.shape[0]
        print(i, end='\r')

    return running_D_loss / (i+1), running_F_loss / (i+1), total_hit / total_num

def do_semi_dataloader(source_dataloader, test_dataloader, threshold=0.5, select_num=1000):
    prediction = []
    target_path = [p for p,_ in test_dataloader.dataset.samples]
    feature_extractor.eval()
    label_predictor.eval()
    cnt = 0
    softmax = nn.Softmax(dim=-1)

    for target_data, _ in test_dataloader:
        target_data = target_data.cuda()
        with torch.no_grad():  
            class_logits = label_predictor(feature_extractor(target_data))        
            probs = softmax(class_logits)
        prediction.append(probs)

    predictions = torch.cat(prediction, dim=0)
    values, targets = torch.max(predictions, dim=1)

    # print(targets[:100])
    # print('target_path', target_path[:100])
    # valid_indices = values > threshold
    # valid_indices = [i for i, x in enumerate(valid_indices) if x]

    values_sorted, values_indices = torch.sort(values)

    targets_sorted = targets[values_indices]
    target_path_sorted = [target_path[idx] for idx in values_indices]

    select_indices = []
    each_class_num = {}

    for k in range(10):
        valid_indices = (values_sorted > threshold) & (targets_sorted == k)
        valid_indices = [i for i, x in enumerate(valid_indices) if x]

        if len(valid_indices) >= select_num:
            valid_indices = valid_indices[:select_num]
        else:
            if len(valid_indices) > 0:
                valid_indices = valid_indices * (select_num // len(valid_indices))
            else:
                print('len(valid_indices) == 0', k, len(valid_indices))
                valid_indices = []

        select_indices += valid_indices
        each_class_num[k] = len(valid_indices)

    source_dataset_semi = ImageFolder('real_or_drawing/train_data', transform=target_transform)
    source_dataset_semi.samples = []
    for index in select_indices:
        source_dataset_semi.samples.append([target_path_sorted[index], targets_sorted[index].item()]) 
    print('source_dataset_semi data num:', len(source_dataset_semi))
    print('semi data samples', source_dataset_semi.samples[:10])

    return DataLoader(source_dataset_semi, batch_size=args.semi_batch_size, shuffle=True, num_workers=1)

# train 200 epochs
for epoch in range(start_epoch, total_epoch):
    # You should chooose lamnda cleverly.
    lamb = (2 / (1 + math.exp(-lamb_decay * epoch / total_epoch)) - 1) * 1

    if args.do_semi == 1:
        if target_semi_dataloader:
            train_D_loss, train_F_loss, train_acc = train_semi_epoch(source_dataloader, target_dataloader, target_semi_dataloader, lamb=lamb)
        else:
            train_D_loss, train_F_loss, train_acc = train_epoch(source_dataloader, target_dataloader, lamb=lamb)
        
        if epoch % (args.semi_epoch) == 0 and epoch > args.semi_init_epoch:
            target_semi_dataloader = do_semi_dataloader(source_dataloader, test_dataloader, threshold=args.threshold, select_num=args.select_num)
    elif args.do_semi == 0:
        train_D_loss, train_F_loss, train_acc = train_epoch(source_dataloader, target_dataloader, lamb=lamb)
    

    torch.save(feature_extractor.state_dict(), os.path.join(log_path, 'feature_extractor.bin'))
    torch.save(label_predictor.state_dict(), os.path.join(log_path, 'label_predictor.bin'))
    torch.save(domain_classifier.state_dict(), os.path.join(log_path, 'domain_classifier.bin'))

    print('epoch {:>3d}: train D loss: {:6.4f}, train F loss: {:6.4f}, acc {:6.4f}'.format(epoch, train_D_loss, train_F_loss, train_acc))


# # Inference
# 
# 就跟前幾次作業一樣。這裡我使用pd來生產csv，因為看起來比較潮(?)
# 
# 此外，200 epochs的Accuracy可能會不太穩定，可以多丟幾次或train久一點。

# In[ ]:


result = []
label_predictor.eval()
feature_extractor.eval()
for i, (test_data, _) in enumerate(test_dataloader):
    test_data = test_data.cuda()

    class_logits = label_predictor(feature_extractor(test_data))

    x = torch.argmax(class_logits, dim=1).cpu().detach().numpy()
    result.append(x)

import pandas as pd
result = np.concatenate(result)

# Generate your submission
df = pd.DataFrame({'id': np.arange(0,len(result)), 'label': result})
df.to_csv(os.path.join(log_path, 'DaNN_submission.csv'),index=False)


# # Training Statistics
# 
# - Number of parameters:
#   - Feature Extractor: 2, 142, 336
#   - Label Predictor: 530, 442
#   - Domain Classifier: 1, 055, 233
# 
# - Simple
#  - Training time on colab: ~ 1 hr
# - Medium
#  - Training time on colab: 2 ~ 4 hr
# - Strong
#  - Training time on colab: 5 ~ 6 hrs
# - Boss
#  - **Unmeasurable**

# # Learning Curve (Strong Baseline)
# * This method is slightly different from colab.
# 
# ![Loss Curve](https://i.imgur.com/vIujQyo.png)
# 
# # Accuracy Curve (Strong Baseline)
# * Note that you cannot access testing accuracy. But this plot tells you that even though the model overfits the training data, the testing accuracy is still improving, and that's why you need to train more epochs.
# 
# ![Acc Curve](https://i.imgur.com/4W1otXG.png)
# 
# 

# # Q&A
# 
# 有任何問題 Domain Adaptation 的問題可以寄信到ntu-ml-2021spring-ta@googlegroups.com。
# 
# 時間允許的話我會更新在這裡。
# 
# # Special Thanks
# 這次的作業其實是我出在 2019FALL 的 ML Final Project，以下是我認為在 Final Report 不錯的幾組，有興趣的話歡迎大家參考看看。
# 
# [NTU_r08942071_太神啦 / 組長: 劉正仁同學](https://drive.google.com/open?id=11uNDcz7_eMS8dMQxvnWsbrdguu9k4c-c)
# 
# [NTU_r08921a08_CAT / 組長: 廖子毅同學](https://drive.google.com/open?id=1xIkSs8HAShdcfV1E0NEnf4JDbL7POZTf)
# 

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

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

test_dataset = ImageFolder('real_or_drawing/test_data', transform=test_transform)

print(test_dataset)
test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=1)

resume = 'record/DaNN2-1-3-changetest-originadam-15000-ls'

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

feature_extractor = FeatureExtractor().cuda()

label_predictor = LabelPredictor().cuda()
domain_classifier = DomainClassifier().cuda()

ckpt_path = os.path.join(resume, 'feature_extractor.bin')
state_dict = torch.load(ckpt_path)
feature_extractor.load_state_dict(state_dict)

ckpt_path = os.path.join(resume, 'label_predictor.bin')
state_dict = torch.load(ckpt_path)
label_predictor.load_state_dict(state_dict)

ckpt_path = os.path.join(resume, 'domain_classifier.bin')
state_dict = torch.load(ckpt_path)
domain_classifier.load_state_dict(state_dict)



result = []
result_prob = []
label_predictor.eval()
feature_extractor.eval()
softmax = nn.Softmax(dim=-1)

for i, (test_data, _) in enumerate(test_dataloader):
    test_data = test_data.cuda()

    class_logits = label_predictor(feature_extractor(test_data))   
    probs = softmax(class_logits)

    x = torch.argmax(class_logits, dim=1).cpu().detach().numpy()
    p = torch.max(probs, dim=1)[0].cpu().detach().numpy()
    result.append(x)
    result_prob.append(p)

import pandas as pd
result = np.concatenate(result)
result_prob = np.concatenate(result_prob)

# Generate your submission
df = pd.DataFrame({'id': np.arange(0,len(result)), 'label': result, 'prob': result_prob})
df.to_csv(os.path.join('DaNN_submission_prob.csv'),index=False)

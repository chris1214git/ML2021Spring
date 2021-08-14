# Import necessary packages.
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
# "ConcatDataset" and "Subset" are possibly useful when doing semi-supervised learning.
from torch.utils.data import ConcatDataset, DataLoader, Subset
from torchvision.datasets import DatasetFolder

# This is for the progress bar.
# from tqdm.auto import tqdm # https://discuss.pytorch.org/t/error-while-multiprocessing-in-dataloader/46845/9
from tqdm import tqdm
import argparse
import os, sys
import json
from timm.data.auto_augment import rand_augment_transform
from PIL import Image
from sklearn.metrics import confusion_matrix


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='tf_efficientnet_b5')
parser.add_argument('--epochs', type=int, default=80)
parser.add_argument('--lr', type=float, default=0.0003)
parser.add_argument('--batch_size', type=int, default=128)

parser.add_argument('--dropout', type=float, default=0.)
parser.add_argument('--l1', type=float, default=0.)
parser.add_argument('--l2', type=float, default=0.)
parser.add_argument('--label_smoothing', type=float, default=0.)

parser.add_argument('--augment', type=str, default=None)
parser.add_argument('--augment_param', type=str, default='randaug')
parser.add_argument('--aug_resize', type=int, default=224)

parser.add_argument('--opt', type=str, default='adam', choices=['sgd', 'adam', 'adamw', 'adamp'])
parser.add_argument('--weight_decay', type=float, default=1e-5)
parser.add_argument('--momentum', type=float, default=0.95)

parser.add_argument('--opt_scheduler', type=str, default='none')
parser.add_argument('--opt_scheduler_tinit', type=int, default=50)
parser.add_argument('--opt_scheduler_cycle', type=int, default=2)
parser.add_argument('--opt_scheduler_warmup_t', type=int, default=10)
parser.add_argument('--opt_scheduler_minlr', type=float, default=0.0003)
# https://fastai.github.io/timmdocs/SGDR#cycle_limit

parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--train_all', type=int, default=0)
parser.add_argument('--do_semi', type=int, default=0)
parser.add_argument('--semi_lr', type=float, default=30)
parser.add_argument('--threshold', type=float, default=0.65)
parser.add_argument('--select_unlabel_num', type=int, default=100)
parser.add_argument('--load_pretrain_model', type=str, default=None)

parser.add_argument('--num_workers', type=int, default=1)
parser.add_argument('--cuda', type=int, default=0)
parser.add_argument('--log_dir', type=str, default='efficientnet_b5_0')
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

# os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)
import torch
print(args.cuda)
torch.cuda.set_device(args.cuda)
print('cuda:', torch.cuda.current_device())
device = "cuda" if torch.cuda.is_available() else "cpu"

img_size = args.aug_resize

# https://github.com/ildoonet/pytorch-randaugment
if args.augment == 'randaug':
    from RandAugment import RandAugment
    train_tfm = transforms.Compose([
        transforms.RandomResizedCrop((img_size, img_size)),   
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    train_tfm.transforms.insert(0, RandAugment(1, 2))

    test_tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
elif args.augment is None:        
    train_tfm = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    test_tfm = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
elif args.augment == 'simple':         # https://github.com/Prakhar998/food-101
    
    train_tfm = transforms.Compose([
        transforms.RandomAffine(degrees=30, shear=30),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  
        transforms.RandomResizedCrop((img_size, img_size)),      
        # transforms.RandomResizedCrop(224),      
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    )
    test_tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    )
elif args.augment == 'simple2':         # https://github.com/Prakhar998/food-101
    img_size = args.aug_resize
    train_tfm = transforms.Compose([
        transforms.RandomAffine(degrees=10, shear=10),
        transforms.RandomHorizontalFlip(),
        # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  
        transforms.RandomResizedCrop((img_size, img_size), scale=(0.5, 1.0)),      
        # transforms.RandomResizedCrop(224),      
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    )
    test_tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    )

# https://blog.csdn.net/qq_24739717/article/details/102743691
# RandomAffine,
# ColorJitter,

batch_size = args.batch_size

# Construct datasets.
# The argument "loader" tells how torchvision reads the data.
train_set = DatasetFolder("food-11/training/labeled", loader=lambda x: Image.open(x), extensions="jpg", transform=train_tfm)
valid_set = DatasetFolder("food-11/validation", loader=lambda x: Image.open(x), extensions="jpg", transform=test_tfm)
unlabeled_set_clean = DatasetFolder("food-11/training/unlabeled", loader=lambda x: Image.open(x), extensions="jpg", transform=test_tfm)
unlabeled_set = DatasetFolder("food-11/training/unlabeled", loader=lambda x: Image.open(x), extensions="jpg", transform=train_tfm)
test_set = DatasetFolder("food-11/testing", loader=lambda x: Image.open(x), extensions="jpg", transform=test_tfm)

# Construct data loaders.
if args.train_all == 0:
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
elif args.train_all == 1:
    train_set = ConcatDataset([train_set, valid_set])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    valid_loader = None
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

semi_loader = None

# https://github.com/wangleiofficial/label-smoothing-pytorch
import torch.nn.functional as F

def linear_combination(x, y, epsilon):
    return epsilon * x + (1 - epsilon) * y


def reduce_loss(loss, reduction='mean'):
    return loss.mean() if reduction == 'mean' else loss.sum() if reduction == 'sum' else loss


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, epsilon: float = 0.1, reduction='mean'):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction

    def forward(self, preds, target):
        n = preds.size()[-1]
        log_preds = F.log_softmax(preds, dim=-1)
        loss = reduce_loss(-log_preds.sum(dim=-1), self.reduction)
        nll = F.nll_loss(log_preds, target, reduction=self.reduction)
        return linear_combination(loss / n, nll, self.epsilon)


def get_pseudo_labels(args, model, threshold=0.65):
    # This functions generates pseudo-labels of a dataset using given model.
    # It returns an instance of DatasetFolder containing images whose prediction confidences exceed a given threshold.
    # You are NOT allowed to use any models trained on external data for pseudo-labeling.

    # modified DatasetFolder
    # https://github.com/pytorch/vision/blob/bb5af1d77658133af8be8c9b1a13139722315c3a/torchvision/datasets/folder.py#L93
    device = "cuda" if torch.cuda.is_available() else "cpu"

    unlabeled_set_clean = DatasetFolder("food-11/training/unlabeled", loader=lambda x: Image.open(x), extensions="jpg", transform=test_tfm)
    unlabeled_set = DatasetFolder("food-11/training/unlabeled", loader=lambda x: Image.open(x), extensions="jpg", transform=test_tfm)
    select_num = args.select_unlabel_num
    # Make sure the model is in eval mode.
    model.eval()
    # Define softmax function.
    softmax = nn.Softmax(dim=-1)
    dataloader = DataLoader(unlabeled_set_clean, batch_size=128, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    
    predictions = []
    # Iterate over the dataset by batches.
    for batch in tqdm(dataloader):
        img, _ = batch

        # Forward the data
        # Using torch.no_grad() accelerates the forward process.
        with torch.no_grad():
            logits = model(img.to(device))

        # Obtain the probability distributions by applying softmax on logits.
        probs = softmax(logits)
        predictions.append(probs)
        # ---------- TODO ----------
        # Filter the data and construct a new dataset.
    
    predictions = torch.cat(predictions, 0) 
    values, targets = torch.max(predictions, dim=1)
    valid_indices = values > threshold
    valid_indices = [i for i, x in enumerate(valid_indices) if x]

    values_sorted, values_indices = torch.sort(values)
    targets_sorted = targets[values_indices]

    # print('valid_indices', valid_indices)
    # print('targets', targets)
    select_indices = []
    each_class_num = {}

    for k in range(11):
        valid_indices = (values_sorted > threshold) & (targets_sorted == k)
        # print('valid_indices', valid_indices)
        valid_indices = [i for i, x in enumerate(valid_indices) if x]

        if len(valid_indices) >= select_num:
            valid_indices = valid_indices[:select_num]
        else:
            if len(valid_indices) > 0:
                valid_indices = valid_indices * (select_num // len(valid_indices))
            else:
                print('len(valid_indices) == 0', len(valid_indices))
                valid_indices = []

        select_indices += valid_indices
        each_class_num[k] = len(valid_indices)
    # targets_k, targets_counts = torch.unique(targets[valid_indices], return_counts=True)
    # print(targets_k, targets_counts)

    unlabeled_set.samples = [(unlabeled_set.samples[i][0], targets[i].item()) for i in select_indices]

    print('each class num', each_class_num)
    # print('valid threshold', threshold)
    print('unlabeled_set.samples len', len(unlabeled_set.samples))
    # # Turn off the eval mode.
    model.train()
    return unlabeled_set

def get_pseudo_labels2(args, model, threshold=0.65):
    # This functions generates pseudo-labels of a dataset using given model.
    # It returns an instance of DatasetFolder containing images whose prediction confidences exceed a given threshold.
    # You are NOT allowed to use any models trained on external data for pseudo-labeling.

    # modified DatasetFolder
    # https://github.com/pytorch/vision/blob/bb5af1d77658133af8be8c9b1a13139722315c3a/torchvision/datasets/folder.py#L93
    device = "cuda" if torch.cuda.is_available() else "cpu"

    unlabeled_set_clean = DatasetFolder("food-11/training/unlabeled", loader=lambda x: Image.open(x), extensions="jpg", transform=test_tfm)
    unlabeled_set = DatasetFolder("food-11/training/unlabeled", loader=lambda x: Image.open(x), extensions="jpg", transform=test_tfm)
    select_num = args.select_unlabel_num
    # Make sure the model is in eval mode.
    model.eval()
    # Define softmax function.
    softmax = nn.Softmax(dim=-1)
    dataloader = DataLoader(unlabeled_set_clean, batch_size=128, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    
    predictions = []
    # Iterate over the dataset by batches.
    for batch in tqdm(dataloader):
        img, _ = batch

        # Forward the data
        # Using torch.no_grad() accelerates the forward process.
        with torch.no_grad():
            logits = model(img.to(device))

        # Obtain the probability distributions by applying softmax on logits.
        probs = softmax(logits)
        predictions.append(probs)
        # ---------- TODO ----------
        # Filter the data and construct a new dataset.
    
    predictions = torch.cat(predictions, 0) 
    values, targets = torch.max(predictions, dim=1)
    valid_indices = values > threshold
    valid_indices = [i for i, x in enumerate(valid_indices) if x]

    values_sorted, values_indices = torch.sort(values)
    targets_sorted = targets[values_indices]

    # print('valid_indices', valid_indices)
    # print('targets', targets)
    select_indices = []
    each_class_num = {}

    for k in range(11):
        valid_indices = (values_sorted > threshold) & (targets_sorted == k)
        # print('valid_indices', valid_indices)
        valid_indices = [i for i, x in enumerate(valid_indices) if x]

        if len(valid_indices) >= select_num:
            valid_indices = valid_indices[:select_num]
        else:
            if len(valid_indices) > 0:
                valid_indices = valid_indices * (select_num // len(valid_indices))
            else:
                print('len(valid_indices) == 0', len(valid_indices))
                valid_indices = []

        select_indices += valid_indices
        each_class_num[k] = len(valid_indices)
    # targets_k, targets_counts = torch.unique(targets[valid_indices], return_counts=True)
    # print(targets_k, targets_counts)

    unlabeled_set.samples = [(unlabeled_set.samples[i][0], predictions[i].cpu().numpy()) for i in select_indices]

    print('each class num', each_class_num)
    # print('valid threshold', threshold)
    print('unlabeled_set.samples len', len(unlabeled_set.samples))
    # # Turn off the eval mode.
    model.train()
    return unlabeled_set

# https://fastai.github.io/timmdocs/
import timm

if args.load_pretrain_model == 'tf_efficientnet_b5_lr4':
    with open(os.path.join('record', args.load_pretrain_model, 'para.txt')) as f:
        data = f.read()
    setting = json.loads(data)    
    args.model = setting['model']
    model = timm.create_model(args.model, pretrained=False, num_classes=11)
    # add dropout
    model.load_state_dict(torch.load(os.path.join('record', args.load_pretrain_model, 'model.ckpt')))
    model.classifier = nn.Sequential(
         nn.Dropout(args.dropout),
         nn.Linear(model.classifier.in_features, 11)
    )    
elif args.load_pretrain_model:
    with open(os.path.join('record', args.load_pretrain_model, 'para.txt')) as f:
        data = f.read()
    setting = json.loads(data)    
    args.model = setting['model']
    model = timm.create_model(args.model, pretrained=False, num_classes=11)
    # add dropout
    model.classifier = nn.Sequential(
         nn.Dropout(args.dropout),
         nn.Linear(model.classifier.in_features, 11)
    )    
    model.load_state_dict(torch.load(os.path.join('record', args.load_pretrain_model, 'model.ckpt'), map_location=device))
else:
    model = timm.create_model(args.model, pretrained=False, num_classes=11)
    # add dropout
    model.classifier = nn.Sequential(
         nn.Dropout(args.dropout),
         nn.Linear(model.classifier.in_features, 11)
    )
model.device = device


model = model.to(device)
# For the classification task, we use cross-entropy as the measurement of performance.
criterion = nn.CrossEntropyLoss()
semi_criterion = nn.KLDivLoss()
semi_criterion = nn.MSELoss()

if args.label_smoothing > 0:
    criterion = LabelSmoothingCrossEntropy(args.label_smoothing)

from timm.optim.optim_factory import create_optimizer
optimizer = create_optimizer(args, model)

if args.opt_scheduler == 'cos':
    from timm.scheduler.cosine_lr import CosineLRScheduler
    scheduler = CosineLRScheduler(optimizer, t_initial=args.opt_scheduler_tinit, cycle_limit=args.opt_scheduler_cycle,\
                                     warmup_t=args.opt_scheduler_warmup_t, decay_rate=1., t_mul=1, lr_min=args.opt_scheduler_minlr, warmup_lr_init=1e-5)
    def get_lr_per_epoch(scheduler, num_epoch):
        lr_per_epoch = []
        for epoch in range(num_epoch):
            lr_per_epoch.append(scheduler.get_epoch_values(epoch))
        return lr_per_epoch
    lr_per_epoch = get_lr_per_epoch(scheduler, 50*2)
    import matplotlib.pyplot as plt
    plt.plot(lr_per_epoch)
    plt.savefig(os.path.join(log_path, 'lr.png'))
else:
    scheduler = None

# The number of training epochs.
n_epochs = args.epochs if scheduler is None else args.opt_scheduler_tinit * args.opt_scheduler_cycle
do_semi = args.do_semi
# the path where checkpoint saved
model_path = os.path.join(log_path, './model.ckpt')

train_history = {'epoch':[], 'loss':[], 'accuracy':[]}
valid_history = {'epoch':[], 'loss':[], 'accuracy':[]}

model_para_num = sum(p.numel() for p in model.parameters() if p.requires_grad)

# print(model)
print('model', args.model)
print('model parameters(M)', model_para_num/1000_000)

print('optimizer', optimizer)
print('scheduler', scheduler)
print('criterion', criterion)

valid_loader.dataset.targets
best_acc = 0.0
for epoch in range(n_epochs):
    # ---------- TODO ----------
    # In each epoch, relabel the unlabeled dataset for semi-supervised learning.
    # Then you can combine the labeled dataset and pseudo-labeled dataset for the training.
    if do_semi == 1 and epoch % 10 == 0:
        # Obtain pseudo-labels for unlabeled data using trained model.
        pseudo_set = get_pseudo_labels2(args, model, threshold=args.threshold)

        # Construct a new dataset and a data loader for training.
        # This is used in semi-supervised learning only.
        # concat_dataset = ConcatDataset([train_set, pseudo_set])

        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
        semi_loader = DataLoader(pseudo_set, batch_size=batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

        # train_loader = DataLoader(concat_dataset, batch_size=batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    # ---------- Training ----------
    # Make sure the model is in train mode before training.
    model.train()

    if semi_loader is not None:
        for batch in tqdm(semi_loader):

            # A batch consists of image data and corresponding labels.
            imgs, labels = batch

            # Forward the data. (Make sure data and model are on the same device.)
            logits = model(imgs.to(device))
            # print('logits', logits.shape)
            # print('labels', labels.shape)

            # Calculate the cross-entropy loss.
            # We don't need to apply softmax before computing cross-entropy as it is done automatically.
            loss = semi_criterion(logits, labels.to(device))
            loss = loss * args.semi_lr
            # all_params = torch.cat([x.view(-1) for x in model.parameters()])
            # l1_regularization = args.l1 * torch.norm(all_params, 1)
            # l2_regularization = args.l2 * torch.norm(all_params, 2)
            # loss = loss + l1_regularization + l2_regularization
            
            # Gradients stored in the parameters in the previous step should be cleared out first.
            optimizer.zero_grad()

            # Compute the gradients for parameters.
            loss.backward()

            # Clip the gradient norms for stable training.
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)

            # Update the parameters with computed gradients.
            optimizer.step()

    # These are used to record information in training.
    train_loss = []
    train_accs = []
    # Iterate the training set by batches.
    for batch in tqdm(train_loader):

        # A batch consists of image data and corresponding labels.
        imgs, labels = batch

        # Forward the data. (Make sure data and model are on the same device.)
        logits = model(imgs.to(device))

        # Calculate the cross-entropy loss.
        # We don't need to apply softmax before computing cross-entropy as it is done automatically.
        loss = criterion(logits, labels.to(device))

        all_params = torch.cat([x.view(-1) for x in model.parameters()])
        l1_regularization = args.l1 * torch.norm(all_params, 1)
        l2_regularization = args.l2 * torch.norm(all_params, 2)
        loss = loss + l1_regularization + l2_regularization
        
        # Gradients stored in the parameters in the previous step should be cleared out first.
        optimizer.zero_grad()

        # Compute the gradients for parameters.
        loss.backward()

        # Clip the gradient norms for stable training.
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)

        # Update the parameters with computed gradients.
        optimizer.step()

        # Compute the accuracy for current batch.
        acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

        # Record the loss and accuracy.
        train_loss.append(loss.item())
        train_accs.append(acc.item())



    if scheduler is not None:
        scheduler.step(epoch)

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
    model.eval()

    # These are used to record information in validation.
    valid_loss = []
    valid_accs = []
    # all_preds = torch.tensor([]).to(device)
    # Iterate the validation set by batches.
    for batch in tqdm(valid_loader):

        # A batch consists of image data and corresponding labels.
        imgs, labels = batch

        # We don't need gradient in validation.
        # Using torch.no_grad() accelerates the forward process.
        with torch.no_grad():
          logits = model(imgs.to(device))

        # We can still compute the loss (but not the gradient).
        loss = criterion(logits, labels.to(device))

        # Compute the accuracy for current batch.
        acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

        # Record the loss and accuracy.
        valid_loss.append(loss.item())
        valid_accs.append(acc.item())
        # all_preds = torch.cat(
        #     (all_preds, logits)
        #     ,dim=0
        # )
    
    # cm = confusion_matrix(valid_loader.dataset.samples, all_preds.argmax(dim=1).cpu())
    # The average loss and accuracy for entire validation set is the average of the recorded values.
    valid_loss = sum(valid_loss) / len(valid_loss)
    valid_acc = sum(valid_accs) / len(valid_accs)

    # Print the information.
    print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")
    # print("Valid confusion_matrix", cm)
    valid_history['epoch'].append(epoch)
    valid_history['loss'].append(valid_loss)
    valid_history['accuracy'].append(valid_acc)

    if valid_acc > best_acc:
        best_acc = valid_acc
        torch.save(model.state_dict(), model_path)
        print('saving model with acc {:.3f}'.format(best_acc))
    # if not validating, save the last epoch
    # if len(val_set) == 0:
    #     torch.save(model.state_dict(), model_path)
    #     print('saving model at last epoch')

    # plot history
    train_history_df = pd.DataFrame(train_history)
    valid_history_df = pd.DataFrame(valid_history)
    import plotly.express as px
    fig = px.line(train_history_df, x="epoch", y="accuracy", title='train accuracy')
    fig.write_image(os.path.join(log_path, 'train_history.png'))

    fig = px.line(valid_history_df, x="epoch", y="accuracy", title='valid accuracy')
    fig.write_image(os.path.join(log_path, 'valid_history.png'))

    train_history_df.to_csv(os.path.join(log_path, 'train_history.csv'), index=False)
    valid_history_df.to_csv(os.path.join(log_path, 'valid_history.csv'), index=False)

"""## **Testing** ##"""
model.eval()

predictions = []

# Iterate the testing set by batches.
for batch in tqdm(test_loader):
    imgs, labels = batch
    with torch.no_grad():
        logits = model(imgs.to(device))
    predictions.extend(logits.argmax(dim=-1).cpu().numpy().tolist())

# Save predictions into the file.
with open(os.path.join(log_path, 'prediction.csv'), "w") as f:
    # The first row must be "Id, Category"
    f.write("Id,Category\n")
    # For the rest of the rows, each image id corresponds to a predicted class.
    for i, pred in  enumerate(predictions):
         f.write(f"{i},{pred}\n")

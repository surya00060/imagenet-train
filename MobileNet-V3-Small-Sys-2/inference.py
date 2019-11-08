from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.datasets as datasets
import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models.mobilenetv3 import MobileNetV3
from models.mobilenetv3 import BottleNeck_Sys_Friendly

from utils import progress_bar


torch.manual_seed(42)

parser = argparse.ArgumentParser(description='PyTorch IMAGENET Training')
#parser.add_argument('--lr', default=0.0005, type=float, help='learning rate')
#parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
traindir = '/scratch/ImageNet-Train-Surya'
valdir   = '/scratch/ImageNet-Val-Surya'
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

#train_dataset = torchvision.datasets.ImageNet(rootdir, split='train', download=True)
#val_dataset = torchvision.datasets.ImageNet(rootdir, split='val', download=True)
#train_dataset = torchvision.datasets.ImageNet(rootdir, split='train', download=True, transforms.Compose([transforms.RandomResizedCrop(input_size),transforms.RandomHorizontalFlip(),transforms.ToTensor(),normalize]))
# val_dataset = torchvision.datasets.ImageNet(rootdir, split='val', download=True,transforms.Compose([transforms.Resize(int(input_size/0.875)),transforms.CenterCrop(input_size),transforms.ToTensor(),normalize,])

# train_dataset = datasets.ImageFolder(traindir,transforms.Compose([transforms.RandomResizedCrop(224),transforms.RandomHorizontalFlip(),transforms.ToTensor(),normalize]))
# train_loader  = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True,num_workers=4, pin_memory=True)

val_dataset   = datasets.ImageFolder(valdir, transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor(),normalize]))
val_loader    = torch.utils.data.DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)

print('==> Building model..')
net = MobileNetV3(mode='small')
state_dict = torch.load('mobilenetv3_small_67.4.pth.tar')
net.load_state_dict(state_dict, strict=True)

# l = []
# for name, child in net.named_children():
#     if name == 'features':
#         for x,y in child.named_children():
#             if x == '11':
#                 l.append(BottleNeck_Sys_Friendly(96, 96, 5, 1, 576, True, 'HS'))
#             else:
#                 l.append(y)
#     else:
#         l.append(child)
    
# new_model = nn.Sequential(*l)
# for name, child in new_model.named_children():
#        	if name == '11':
#             for param in child.parameters():
#                 param.requires_grad = True
#         else:
#             for param in child.parameters():
#                 param.requires_grad = False


# net = new_model
net = net.to(device)
if device == 'cuda':
     net = torch.nn.DataParallel(net)
     cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
def test(epoch):
     net.eval()
     test_loss = 0
     correct = 0
     total = 0
     with torch.no_grad():
         for batch_idx, (inputs, targets) in enumerate(val_loader):
             inputs, targets = inputs.to(device), targets.to(device)
             outputs = net(inputs)
             loss = criterion(outputs, targets)

             test_loss += loss.item()
             _, predicted = outputs.max(1)
             total += targets.size(0)
             correct += predicted.eq(targets).sum().item()

             progress_bar(batch_idx, len(val_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                 % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))


test(0)

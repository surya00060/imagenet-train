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
from models.mobilenetv3 import Flatten

from utils import progress_bar
from folder2lmdb import ImageFolderLMDB

torch.manual_seed(42)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
#torch.cuda.set_device(opt.gpu_ids[4])
parser = argparse.ArgumentParser(description='PyTorch IMAGENET Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
traindir = '/media/iitm/data1/Surya/train.lmdb'
valdir   = '/media/iitm/data1/Surya/val.lmdb'
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])



print('==> Building model..')
net = MobileNetV3(mode='small')
state_dict = torch.load('mobilenetv3_small_67.4.pth.tar')
net.load_state_dict(state_dict, strict=True)

l = []
for name, child in net.named_children():
    if name == 'features':
        for x,y in child.named_children():
            if x == '11':
                l.append(BottleNeck_Sys_Friendly(96, 96, 5, 1, 576, True, 'HS'))
            else:
                l.append(y)
    elif name == 'classifier':
        l.append(Flatten())
        l.append(child)

flag = False
new_model = nn.Sequential(*l)
for name, child in new_model.named_children():
       	if name == '11':
            for param in child.parameters():
                param.requires_grad = True
            flag = True
        else:
            if flag == False:
                for param in child.parameters():
                    param.requires_grad = False


net = new_model
net = net.to(device)
if device == 'cuda':
     #net = torch.nn.DataParallel(net)
     cudnn.benchmark = True

if args.resume:
     # Load checkpoint.
     print('==> Resuming from checkpoint..')
     assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
     checkpoint = torch.load('./checkpoint/BestModel.t7')
     net.load_state_dict(checkpoint['net'])
     best_acc = checkpoint['acc']
     start_epoch = checkpoint['epoch']
     initial_learning_rate = checkpoint['initial_lr']


criterion = nn.CrossEntropyLoss()
optimizer = optim.RMSprop([{'params':net[11].parameters(), 'lr': 0.001},{'params':net[12].parameters()},{'params':net[13].parameters()},{'params':net[14].parameters()},{'params':net[15].parameters()},{'params':net[16].parameters()},{'params':net[17].parameters()}], lr=0.0005, alpha=0.99,momentum=0.9,weight_decay = 1e-5)
#optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, momentum=0.9, weight_decay=4e-5)
#scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 159, 0.00001)


      #lr= args.lr * (0.01 ** (epoch//3))
i = 0
lr= 0.0005 * (0.01 ** (100//10))
optimizer.param_groups[1:7]['lr'] = lr
optimizer.param_groups[1]['lr'] = lr
optimizer.param_groups[2]['lr'] = lr
optimizer.param_groups[3]['lr'] = lr
optimizer.param_groups[4]['lr'] = lr
optimizer.param_groups[5]['lr'] = lr
optimizer.param_groups[6]['lr'] = lr
for param_group in optimizer.param_groups:
      print(param_group['lr'], optimizer.param_groups[i]['lr'])
      i = i + 1


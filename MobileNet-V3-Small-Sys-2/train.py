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
from torch.utils.tensorboard import SummaryWriter

import os
import argparse

from models.mobilenetv3 import MobileNetV3
from models.mobilenetv3 import BottleNeck_Sys_Friendly
from models.mobilenetv3 import Flatten

from utils import progress_bar
from folder2lmdb import ImageFolderLMDB

torch.manual_seed(42)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
parser = argparse.ArgumentParser(description='PyTorch IMAGENET Training')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

#-------------------------------------------------------------------------------------------------------
print('==> Preparing data..')
traindir = '/media/iitm/data1/Surya/train.lmdb'
valdir   = '/media/iitm/data1/Surya/val.lmdb'
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])


train_transform = transforms.Compose([transforms.RandomResizedCrop(224),transforms.RandomHorizontalFlip(),transforms.ToTensor(),normalize])
train_dataset = ImageFolderLMDB(traindir, train_transform)
train_loader  = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True,num_workers=4, pin_memory=True)

val_transform = transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor(),normalize])
val_dataset   = ImageFolderLMDB(valdir, val_transform)
val_loader    = torch.utils.data.DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=2, pin_memory=True)


writer = SummaryWriter()
lrwriter = SummaryWriter()

#------------------------------------------------------------------------------------------------------------
print('==> Building model..')
net = MobileNetV3(mode='small', dropout=0)
# state_dict = torch.load('mobilenetv3_small_67.4.pth.tar')
# net.load_state_dict(state_dict, strict=True)

# l = []
# for name, child in net.named_children():
#     if name == 'features':
#         for x,y in child.named_children():
#             if x == '11':
#                 l.append(BottleNeck_Sys_Friendly(96, 96, 5, 1, 576, True, 'HS'))
#             else:
#                 l.append(y)
#     elif name == 'classifier':
#         l.append(Flatten())
#         l.append(child)

# flag = False
# new_model = nn.Sequential(*l)
# net = new_model


net = net.to(device)
if device == 'cuda':
     #net = torch.nn.DataParallel(net, device_ids=[0,2,3,4])
     cudnn.benchmark = True

if args.resume:
     # Load checkpoint.
     print('==> Resuming from checkpoint..')
     assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
     checkpoint = torch.load('./checkpoint/BestModel.t7')
     net.load_state_dict(checkpoint['net'])
     best_acc = checkpoint['acc']
     start_epoch = checkpoint['epoch']
#---------------------------------------------------------------------------------------------
print('==> Hyperparameter Tuning..')
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-5)
#optimizer = optim.Adam(net.parameters(), lr=3e-4, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5, amsgrad=False)
#optimizer = optim.RMSprop(net.parameters(), lr=args.lr, alpha=0.99,momentum=0.9,weight_decay = 1e-5)
#------------------------------------------------------------------------------------------------



def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        progress_bar(batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'% (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

    writer.add_scalar('LossvsEpoch', train_loss, epoch)
    


def test(epoch):
    global best_acc
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

            progress_bar(batch_idx, len(val_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'% (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    
    writer.add_scalar('TestAccuracyvsEpoch', (1.0*correct)/total, epoch)
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving Best Model..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/BestModel.t7')
        best_acc = acc
    print('Saving Last Model..')
    state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
            }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(state, './checkpoint/LastEpoch.t7')

#------------------------------------------------------------------
print('==> Started Training model..')
for epoch in range(start_epoch, start_epoch+50):
      lrwriter.add_scalar('LRvsEpoch', optimizer.param_groups[0]['lr'], epoch)
      train(epoch)
      test(epoch)

writer.close()
lrwriter.close()
#---------------------------------------

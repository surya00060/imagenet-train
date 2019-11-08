'''Squueze Net in Pytorch.'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class fire(nn.Module):
    def __init__(self, inplanes, squeeze_planes, expand_planes):
        super(fire, self).__init__()
        squeeze_planes = expand_planes//2
        self.conv1 = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm2d(squeeze_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(squeeze_planes, expand_planes, kernel_size=1, stride=1)
        self.bn2 = nn.BatchNorm2d(expand_planes)
        self.conv3 = nn.Conv2d(squeeze_planes, squeeze_planes, kernel_size=(1,3), padding = (0,1), groups=squeeze_planes)
        self.bn3 = nn.BatchNorm2d(squeeze_planes)
        self.conv4 = nn.Conv2d(squeeze_planes, squeeze_planes, kernel_size=(3,1), padding = (1,0), groups=squeeze_planes)
        self.bn4 = nn.BatchNorm2d(squeeze_planes)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        out1 = self.conv2(x)
        out1 = self.bn2(out1)
        out2 = self.conv3(x)
        out2 = self.bn3(out2)
        out3 = self.conv4(x)
        out3 = self.bn4(out3)
        out2 = torch.cat([out2, out3],1)
        out = torch.cat([out1, out2], 1)
        out = self.relu2(out)
        return out


class SqueezeNet(nn.Module):
    def __init__(self):
        super(SqueezeNet, self).__init__()
        #self.conv1 = nn.Conv2d(3, 96, kernel_size=3, stride=1, padding=1) # 32
        #self.bn1 = nn.BatchNorm2d(96)
        #self.relu = nn.ReLU(inplace=True)
        self.dconv1_h = nn.Conv2d(3, 3, kernel_size=(1,3), padding = (0,1),
        stride=2,groups=3)
        self.dconv1_v = nn.Conv2d(3, 3, kernel_size=(3,1), padding =(1,0),
        stride=2,groups=3)
        self.bn1      = nn.BatchNorm2d(6)        
        self.pconv1   = nn.Conv2d(6, 64, kernel_size=1)
        self.bn2      = nn.BatchNorm2d(64)

        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2) # 16
        self.fire2 = fire(64, 16, 64)
        self.fire3 = fire(128, 16, 64)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2) #8
        self.fire4 = fire(128, 32, 128)
        self.fire5 = fire(256, 32, 128)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3,stride=2) #4
        self.fire6 = fire(256, 48, 192)
        self.fire7 = fire(384, 48, 192)
        self.fire8 = fire(384, 64, 256)
        self.fire9 = fire(512, 64, 256)
        
        self.drop = nn.Dropout(p=0.5)
        self.conv2 = nn.Conv2d(512, 1000, kernel_size=1, stride=1)
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        
    def forward(self, x):
        #x = self.conv1(x)
        #x = self.bn1(x)
        #x = self.relu(x)
        x = torch.cat( (self.dconv1_h(x), self.dconv1_v(x)), 1 )
        x = self.bn1(x)
        x = F.relu(self.bn2( self.pconv1(x)))
        #print(x.size())
        x = self.maxpool1(x)
        x = self.fire2(x)
        x = self.fire3(x)
        x = self.fire4(x)
        x = self.maxpool2(x)
        x = self.fire5(x)
        x = self.fire6(x)
        x = self.fire7(x)
        #print(x.size())
        x = self.fire8(x)
        #print(x.size())
        x = self.maxpool3(x)
        #print(x.size())
        x = self.fire9(x)
        
        x = self.drop(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), 1000)
        return x

def test():
    net = SqueezeNet()
    x = torch.randn(2,3,224,224)
    y = net(x)
    print(y.size())

#test()


'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import get_model_complexity_info

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class Block(nn.Module):
    '''Depthwise conv + Pointwise conv'''
    def __init__(self, in_planes, out_planes):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        x = self.bn(self.conv1(x))
        return x

class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Sequential( 
                nn.Linear(512*7*7, 4096),
                nn.ReLU(True), nn.Dropout(), nn.Linear(4096,4096),
                nn.ReLU(True), nn.Dropout(), nn.Linear(4096,1000),
                )

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers.append(Block(in_channels,x))
                in_channels = x
        layers += [nn.AdaptiveAvgPool2d((7,7))]
        return nn.Sequential(*layers)
    


def test():
    net = VGG('VGG16')
    flops, params = get_model_complexity_info(net, (224, 224), as_strings=False, print_per_layer_stat=True)                                                  
    print('Flops:{}'.format(flops))                                             
    print('Params:' + str(params))

test()


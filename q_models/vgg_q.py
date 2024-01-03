import sys
import torch
import torchvision
import torch.optim as optim
import torchvision.transforms as transforms
import torch.nn as nn
import math
from typing import Any, Callable, List, Optional, Type,Union
import torch.nn.functional as F
import numpy
from q_models.quantization import quan_Conv2d,quan_Linear
class VGG_q(nn.Module):
    '''
    VGG model 
    '''
    def __init__(self, n_classes=10,dropout_rate=0.0,features=None):
        super(VGG_q, self).__init__()
        self.features = features
        self.classifier = quan_Linear(512, 10)
        # self.relu_class = nn.ReLU()
         # Initialize weights
        for m in self.modules():
            if isinstance(m, quan_Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()


    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        logits = self.classifier(x)
        # logits = self.relu_class(logits)
        probas = torch.softmax(logits, dim=1)
        return logits
  
def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = quan_Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU()]
            else:
                layers += [conv2d, nn.ReLU()]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 
          512, 512, 512, 512, 'M'],
}

class VGG16_q(VGG_q):
    def __init__(self, n_classes=10,dropout_rate=0.0,features=make_layers(cfg['D'],batch_norm=False)):
        super(VGG16_q,self).__init__(n_classes,dropout_rate,features)
##################################################################

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
class AlexNet_model_q(nn.Module):
    def __init__(self, n_classes=10,dropout_rate=0.0):
        super(AlexNet_model_q, self).__init__()
        self.conv1 =  quan_Conv2d(3, 64, kernel_size=3, stride=2, padding=1)
        self.relu1 = nn.ReLU()
        self.maxpool1 =  nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = quan_Conv2d(64, 192, kernel_size=3, padding=2)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = quan_Conv2d(192, 384, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.conv4 = quan_Conv2d(384, 256, kernel_size=3, padding=1)
        self.relu4 = nn.ReLU()
        self.conv5 = quan_Conv2d(256, 256, kernel_size=3, padding=1)
        self.relu5 = nn.ReLU()
        self.maxpool3 =  nn.MaxPool2d(kernel_size=2)
        # self.dropout = nn.Dropout(dropout_rate)
        self.linear1 = quan_Linear(256 * 2 * 2, 4096)
        self.relu6 = nn.ReLU()
        self.linear2 = quan_Linear(4096, 4096)
        self.relu7= nn.ReLU()
        self.linear3 = quan_Linear(4096, n_classes)
        self.relu8= nn.ReLU()

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.maxpool1(x)
        x = self.relu2(self.conv2(x))
        x = self.maxpool2(x)
        x = self.relu3(self.conv3(x))
        x = self.relu4(self.conv4(x))
        x = self.relu5(self.conv5(x))
        x = self.maxpool3(x)
        x = x.view(x.size(0), -1)
        # x = self.dropout(x)
        x = self.relu6(self.linear1(x))
        # x = self.dropout(x)
        x = self.relu7(self.linear2(x))
        x = self.linear3(x)
        x = self.relu8(x) # add only for infreance
        return x        
    def extract_feature(self, x):
        x = self.conv1(x)
        feat1 = self.relu1(x)
        x = self.maxpool1(feat1)
        x = self.conv2(x)
        feat2 = self.relu2(x)
        x = self.maxpool2(feat2)
        x = self.conv3(x)
        feat3 = self.relu3(x)
        x = self.conv4(feat3)
        feat4 = self.relu4(x)
        x = self.conv5(feat4)
        feat5 = self.relu5(x)
        x = self.maxpool3(feat5)
        x = x.view(x.size(0), -1)
        x = self.linear1(x)
        feat6 = self.relu6(x)
        x = self.linear2(feat6)
        feat7 = self.relu7(x)
        out = self.linear3(x)
        

        return [feat1, feat2, feat3,feat4,feat5,feat6,feat7], out
###########################################



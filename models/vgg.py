import torch.nn as nn
import math
import torch.nn.functional as F
class VGG(nn.Module):
    '''
    VGG model 
    '''
    def __init__(self, n_classes=10,dropout_rate=0.0,features=None):
        super(VGG, self).__init__()
        self.features = features
        self.linear1 = nn.Linear(512, n_classes)
        # self.relu1 = nn.ReLU()
        # self.linear2 = nn.Linear(512, 512)
        # self.relu2_last= nn.ReLU()
        # self.linear3 = nn.Linear(512, n_classes)
        # self.relu = nn.ReLU()
         # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()


    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        logits = self.linear1(x)
        # x = self.relu1(x)
        # x = self.linear2(x)
        # x = self.relu2_last(x)
        # logits = self.linear3(x)
       
        return logits
    def extract_feature(self, x, preReLU=False):

        feat1 = self.features(x)
        x = feat1.view(x.size(0), -1)
        out = self.linear1(x)
        # feat2 = self.relu1(x)
        # x = self.linear2(feat2)
        # feat3 = self.relu2_last(x)
        return feat1, out

        if not preReLU:
            feat1 = F.relu(feat1)

        return [feat1], out

def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
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

class VGG16(VGG):
    def __init__(self, n_classes=10,dropout_rate=0.0,features=make_layers(cfg['D'],batch_norm=False)):
        super(VGG16,self).__init__(n_classes,dropout_rate,features)
##################################################################

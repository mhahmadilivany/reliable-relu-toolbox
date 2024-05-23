import torch.nn as nn
import torch.nn.functional as F
class Lenet(nn.Module):
    def __init__(self,n_classes = 10 , dropout_rate=0.0):
        super(Lenet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(256, 120)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(84, n_classes)

    def forward(self, x):
        y = self.conv1(x)
        y = self.relu1(y)
        y = self.pool1(y)
        y = self.conv2(y)
        y = self.relu2(y)
        y = self.pool2(y)
        y = y.view(y.shape[0], -1)
        y = self.fc1(y)
        y = self.relu3(y)
        y = self.fc2(y)
        y = self.relu4(y)
        y = self.fc3(y)
        return y
    def extract_feature(self, x, preReLU=False):
        feat1 = self.conv1(x)
        x = self.relu1(feat1)
        x = self.pool1(x)
        feat2 = self.conv2(x)
        x = self.relu2(feat2)
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        feat3 = self.fc1(x)
        x = self.relu3(feat3)
        feat4 = self.fc2(x)
        x = self.relu4(feat4)
        out = self.fc3(x)
        if not preReLU:
            feat1 = F.relu(feat1)
            feat2 = F.relu(feat2)
            feat3 = F.relu(feat3)
            feat4 = F.relu(feat4)
        return [feat1, feat2, feat3,feat4], out
class LeNet_cifar(nn.Module):
    def __init__(self,n_classes = 10 , dropout_rate=0.0):
        super(LeNet_cifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.relu2 = nn.ReLU()
        self.fc1   = nn.Linear(16*5*5, 120)
        self.relu3 = nn.ReLU()
        self.fc2   = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.fc3   = nn.Linear(84, n_classes)

    def forward(self, x):
        out = self.relu1(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = self.relu2(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = self.relu3(self.fc1(out))
        out = self.relu4(self.fc2(out))
        out = self.fc3(out)
        return out
    def extract_feature(self, x, preReLU=False):
        feat1 = self.conv1(x)
        x = self.relu1(feat1)
        x = F.max_pool2d(x, 2)
        feat2 = self.conv2(x)
        x = self.relu2(feat2)
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        feat3 = self.fc1(x)
        x = self.relu3(feat3)
        feat4 = self.fc2(x)
        x = self.relu4(feat4)
        out = self.fc3(x)
        if not preReLU:
            feat1 = F.relu(feat1)
            feat2 = F.relu(feat2)
            feat3 = F.relu(feat3)
            feat4 = F.relu(feat4)
        return [feat1, feat2, feat3,feat4], out
import torch
from torch import nn

from deepvac.backbones.weights_init import initWeightsKaiming
from deepvac.backbones.conv_layer import Conv2dBNReLU

from .model_fpn import Resnet18FPN, Resnet50FPN, Mobilenetv3LargeFPN

class DBHead(nn.Module):
    def __init__(self, in_channels, k = 50):
        super(DBHead, self).__init__()
        self.k = k
        self.binarize = nn.Sequential(
            Conv2dBNReLU(in_channels, in_channels // 4, kernel_size=3, padding=1),
            nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 2, 2),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels // 4, 1, 2, 2),
            nn.Sigmoid())

        self.thresh = nn.Sequential(
            Conv2dBNReLU(in_channels, in_channels // 4, kernel_size=3, padding=1),
            nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 2, 2),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels // 4, 1, 2, 2),
            nn.Sigmoid())

        self.binarize.apply(initWeightsKaiming)
        self.thresh.apply(initWeightsKaiming)


    def step_function(self, x, y):
        return torch.reciprocal(1 + torch.exp(-self.k * (x - y)))

    def forward(self, x):
        shrink_maps = self.binarize(x)
        threshold_maps = self.thresh(x)
        if self.training:
            binary_maps = self.step_function(shrink_maps, threshold_maps)
            y = torch.cat((shrink_maps, threshold_maps, binary_maps), dim=1)
        else:
            y = torch.cat((shrink_maps, threshold_maps), dim=1)
        return y

class Resnet18DB(nn.Module):
    def __init__(self):
        super(Resnet18DB, self).__init__()
        self._fea = Resnet18FPN()
        self._db = DBHead(512)

    def forward(self, x):
        return self._db(self._fea(x))

class Mobilenetv3LargeDB(nn.Module):
    def __init__(self):
        super(Mobilenetv3LargeDB, self).__init__()
        self._fea = Mobilenetv3LargeFPN()
        self._db = DBHead(512)

    def forward(self, x):
        return self._db(self._fea(x))

def test():
    net = Resnet18DB()
    x = torch.randn(2,3,640,640)
    y = net(x)
    print(y.size())

#test()

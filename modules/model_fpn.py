import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from deepvac.backbones.mobilenet import MobileNetV3, MobileNetV3Large, ResNet50, ResNet18, Conv2dBNReLU, Concat

class Mobilenetv3LargeBackbone(MobileNetV3Large):
    def __init__(self):
        super(Mobilenetv3LargeBackbone, self).__init__(width_mult=1.)

    def initFc(self):
        self.downsampler_list = [3,6,12]

    def forward(self,x):
        out = []
        for i, fea in enumerate(self.features):
            x = fea(x)
            if i in self.downsampler_list:
                out.append(x)
        x = self.conv(x)
        out.append(x)
        return out

class Resnet50Backbone(ResNet50):
    def __init__(self):
        super(Resnet50Backbone, self).__init__(class_num=0)

    def initFc(self):
        self.downsampler_list = [2,6,12]

    def forward(self, x):
        out = []
        x = self.conv1(x)
        x = self.maxpool(x)
        for i, fea in enumerate(self.layer):
            x = fea(x)
            if i in self.downsampler_list:
                out.append(x)
        out.append(x)
        return out

class Resnet18Backbone(ResNet18):
    def __init__(self):
        super(Resnet18Backbone, self).__init__(class_num=0)

    def initFc(self):
        self.downsampler_list = [1,3,5]

    def forward(self, x):
        out = []
        x = self.conv1(x)
        x = self.maxpool(x)
        for i, fea in enumerate(self.layer):
            x = fea(x)
            if i in self.downsampler_list:
                out.append(x)
        out.append(x)
        return out

class FpnBackBone(nn.Module):
    def init_backbone(self):
        LOG.logE("You must achieve function init_backbone.", exit=True)

    def __init__(self, kernel_num=7):
        super(FpnBackBone, self).__init__()
        self.init_backbone()
        self.toplayer = Conv2dBNReLU(self.out[3],self.conv_out,kernel_size=1,stride=1,padding=0)

        # Lateral layers
        self.latlayer1 = Conv2dBNReLU(self.out[2], self.conv_out, kernel_size=1, stride=1, padding=0)

        self.latlayer2 = Conv2dBNReLU(self.out[1], self.conv_out, kernel_size=1, stride=1, padding=0)

        self.latlayer3 = Conv2dBNReLU(self.out[0], self.conv_out, kernel_size=1, stride=1, padding=0)

        # Smooth layers
        self.smooth1 = nn.Sequential(
            Conv2dBNReLU(self.conv_out, self.conv_out, kernel_size=3, stride=1, padding=1, groups=self.conv_out),
            Conv2dBNReLU(self.conv_out, self.conv_out, kernel_size=1, padding=0, stride=1)
        )

        self.smooth2 = nn.Sequential(
            Conv2dBNReLU(self.conv_out, self.conv_out, kernel_size=3, stride=1, padding=1, groups=self.conv_out),
            Conv2dBNReLU(self.conv_out, self.conv_out, kernel_size=1, padding=0, stride=1)
        )

        self.smooth3 = nn.Sequential(
            Conv2dBNReLU(self.conv_out, self.conv_out, kernel_size=3, stride=1, padding=1, groups=self.conv_out),
            Conv2dBNReLU(self.conv_out, self.conv_out, kernel_size=1, padding=0, stride=1)
        )

        self.cat = Concat()

    def _upsample(self, x, y, scale=1):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H // scale, W // scale), mode='bilinear')

    def _upsample_add(self, x, y):
        return F.interpolate(x, size=y.size()[2:], mode='bilinear') + y

    def _upsample_cat(self, p2, p3, p4, p5):
        h, w = p2.size()[2:]
        p3 = F.interpolate(p3, size=(h, w), mode='bilinear')
        p4 = F.interpolate(p4, size=(h, w), mode='bilinear')
        p5 = F.interpolate(p5, size=(h, w), mode='bilinear')
        return self.cat([p2, p3, p4, p5])

    def forward(self, x):
        c2, c3, c4, c5 = self.backbone(x)
        
        # Head
        p5 = self.toplayer(c5)
        p4 = self._upsample_add(p5, self.latlayer1(c4))
        p4 = self.smooth1(p4)
        p3 = self._upsample_add(p4, self.latlayer2(c3))
        p3 = self.smooth2(p3)
        p2 = self._upsample_add(p3, self.latlayer3(c2))
        p2 = self.smooth3(p2)

        out = self._upsample_cat(p2, p3, p4, p5)
        return out

class Mobilenetv3LargeFPN(FpnBackBone):
    def init_backbone(self):
        self.backbone = Mobilenetv3LargeBackbone()
        self.out = [24, 40, 112, 960]
        self.conv_out = 128

class Resnet50FPN(FpnBackBone):
    def init_backbone(self):
        self.backbone = Resnet50Backbone()
        self.out = [256, 512, 1024, 2048]
        self.conv_out = 256

class Resnet18FPN(FpnBackBone):
    def init_backbone(self):
        self.backbone = Resnet18Backbone()
        self.out = [64, 128, 256, 512]
        self.conv_out = 128

def test():
    net = Resnet18FPN()
    x = torch.randn(2,3,640,640)
    y = net(x)
    print(y.size())

#test()

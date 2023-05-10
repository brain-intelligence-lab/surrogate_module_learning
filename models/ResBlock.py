import torch
import torch.nn as nn
import torch.nn.functional as F
from models.spiking_layer import LIFSpike, singleLIF

class BasicBlock(nn.Module):
    expansion = 1


    def __init__(self, in_planes, planes, stride=1, T=1):
        super(BasicBlock, self).__init__()
        self.act = LIFSpike(T=T)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = nn.Sequential() if (stride == 1 and in_planes == planes) else nn.Sequential(
            nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(planes)
        )

    def forward(self, x):
        shortcut = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += shortcut
        out = self.act(out)
        return out

class RMSBasciblock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, T=1):
        super(RMSBasciblock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.act = LIFSpike(T=T)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        # self.move = LearnableProj(planes)
        self.move = nn.BatchNorm2d(planes)
        self.downsample = nn.Sequential() if stride == 1 else nn.Sequential(
            # nn.AvgPool2d(2),
            LIFSpike(T=T),
            nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(planes)
        )

    def forward(self, x):
        shortcut = self.downsample(x)
        out = self.act(x)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.act(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += shortcut
        out = self.move(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, T=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.act = LIFSpike(T=T)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.downsample = nn.Sequential() if (stride == 1 and in_planes == planes * self.expansion) \
            else nn.Sequential(
            nn.Conv2d(in_planes, planes * self.expansion, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(planes * self.expansion)
        )

    def forward(self, x):
        shortcut = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out += shortcut
        out = self.act(out)
        return out

class MSBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, T=1):
        super(MSBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.act = LIFSpike(T=T)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = nn.Sequential() if stride == 1 else nn.Sequential(
            nn.AvgPool2d(2), # special case for downsampling reference qinhua paper
            self.act, # avoid mutilple in SNN
            nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(planes)
        )

    def forward(self, x):
        shortcut = self.downsample(x)
        out = self.act(x)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.act(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += shortcut
        return out

class MSBottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, T=1):
        super(MSBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.act = LIFSpike(T=T)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.downsample = nn.Sequential() if stride == 1 else nn.Sequential(
            nn.AvgPool2d(2), # TODO: check if this is correct
            self.act,              # special case for downsampling reference qinhua paper
            nn.Conv2d(in_planes, planes * self.expansion, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(planes * self.expansion)
        )

    def forward(self, x):
        shortcut = self.downsample(x)
        out = self.act(x)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.act(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out += shortcut
        return out

class SingleBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, T=1):
        super(SingleBasicBlock, self).__init__()
        self.act1 = singleLIF()
        self.act2 = singleLIF()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = nn.Sequential() if stride == 1 else nn.Sequential(
            nn.Conv2d(in_planes, planes, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(planes)
        )

    def forward(self, x):
        shortcut = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += shortcut
        out = self.act2(out)
        return out

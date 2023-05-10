import torch
import torch.nn as nn
from models.spiking_layer import ExpandTime, LIFSpike, LIAFSpike
from models.ResBlock import BasicBlock, Bottleneck

def Conv7(in_planes, out_planes):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=7, stride=2, padding=3, bias=False),
        nn.BatchNorm2d(out_planes),
        nn.LeakyReLU(0.01, inplace=True),
    )

def Conv5(in_planes, out_planes):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=5, stride=2, padding=2, bias=False),
        nn.BatchNorm2d(out_planes),
        nn.LeakyReLU(0.01, inplace=True),
    )

def Conv3(in_planes, out_planes):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(out_planes),
        nn.LeakyReLU(0.01, inplace=True),
    )

def Conv1(in_planes, out_planes):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(out_planes),
        nn.LeakyReLU(0.01, inplace=True),
    )

def SConv7(in_planes, out_planes, T):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=7, stride=2, padding=3, bias=False),
        nn.BatchNorm2d(out_planes),
        LIFSpike(T=T),
    )

def SConv5(in_planes, out_planes, T):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=5, stride=2, padding=2, bias=False),
        nn.BatchNorm2d(out_planes),
        LIFSpike(T=T),
    )

def SConv3(in_planes, out_planes, T):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(out_planes),
        LIFSpike(T=T),
    )

def SConv1(in_planes, out_planes, T):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(out_planes),
        LIFSpike(T=T),
    )





class SurrogateBlock(nn.Module):
    def __init__(self, kernels, in_channel, out_channel, num_classes, static_input=True, T=1):
        super(SurrogateBlock, self).__init__()
        self.kernels = kernels
        self.T = T
        self.out_channel = out_channel
        self.static_input = static_input
        convs = []
        if in_channel > out_channel:
            convs.append(Conv1(in_channel, out_channel))
            for kernel in kernels:
                if kernel == 7:
                    convs.append(Conv7(out_channel, out_channel))
                elif kernel == 5:
                    convs.append(Conv5(out_channel, out_channel))
                elif kernel == 3:
                    convs.append(Conv3(out_channel, out_channel))
        else:
            first_kernel = kernels[0]
            if first_kernel == 7:
                convs.append(Conv7(in_channel, out_channel))
            elif first_kernel == 5:
                convs.append(Conv5(in_channel, out_channel))
            elif first_kernel == 3:
                convs.append(Conv3(in_channel, out_channel))
            for kernel in kernels[1:]:
                if kernel == 7:
                    convs.append(Conv7(out_channel, out_channel))
                elif kernel == 5:
                    convs.append(Conv5(out_channel, out_channel))
                elif kernel == 3:
                    convs.append(Conv3(out_channel, out_channel))
        # convs.append(Conv1(out_channel, 512))
        self.convs = nn.Sequential(*convs)
        self.prediction = nn.Sequential(
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten(),
            nn.Linear(out_channel * 4, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Linear(512, num_classes),
        )


    def forward(self, x):
        if self.T == 1:
            x = self.convs(x)
            x = self.prediction(x)
        else:
            if self.static_input:
                # B*T, C, H, W -> B, T, C, H, W
                x = x.view(self.T, -1, *x.shape[1:])
                # remove the Time dimension
                x = x.mean(0)
            x = self.convs(x)
            x = self.prediction(x)
        return x


class SpikeSurrogateBlock(nn.Module):
    def __init__(self, kernels, in_channel, out_channel, num_classes, static_input=True, T=1):
        super(SpikeSurrogateBlock, self).__init__()
        self.kernels = kernels
        self.T = T
        self.out_channel = out_channel
        self.static_input = static_input
        convs = []
        if in_channel > out_channel:
            convs.append(SConv1(in_channel, out_channel, T=self.T))
            for kernel in kernels:
                if kernel == 7:
                    convs.append(SConv7(out_channel, out_channel, T=self.T))
                elif kernel == 5:
                    convs.append(SConv5(out_channel, out_channel, T=self.T))
                elif kernel == 3:
                    convs.append(SConv3(out_channel, out_channel, T=self.T))
        else:
            first_kernel = kernels[0]
            if first_kernel == 7:
                convs.append(SConv7(in_channel, out_channel, T=self.T))
            elif first_kernel == 5:
                convs.append(SConv5(in_channel, out_channel, T=self.T))
            elif first_kernel == 3:
                convs.append(SConv3(in_channel, out_channel, T=self.T))
            for kernel in kernels[1:]:
                if kernel == 7:
                    convs.append(SConv7(out_channel, out_channel, T=self.T))
                elif kernel == 5:
                    convs.append(SConv5(out_channel, out_channel, T=self.T))
                elif kernel == 3:
                    convs.append(SConv3(out_channel, out_channel, T=self.T))
        # convs.append(Conv1(out_channel, 512))
        self.convs = nn.Sequential(*convs)
        self.prediction = nn.Sequential(
            # nn.Dropout(0.5),
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten(),
            nn.Linear(out_channel * 4, num_classes),
        )


    def forward(self, x):
        x = self.convs(x)
        x = self.prediction(x)
        if self.T == 1:
            return x
        else:
            # B*T, C, H, W -> B, T, C, H, W
            x = x.view(self.T, -1, *x.shape[1:])
            # remove the Time dimension
            x = x.mean(0)
            return x

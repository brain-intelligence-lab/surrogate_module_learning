import torch
import torch.nn as nn
import torch.nn.functional as F
from models.ResBlock import MSBasicBlock, MSBottleneck
from models.spiking_layer import LIFSpike, ExpandTime
from models.surrogate_block import SurrogateBlock


class MSResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, T=1, zero_init_residual=False):
        super(MSResNet, self).__init__()
        self.in_planes = 64
        self.T = T
        self.expand_time = ExpandTime(T=T)

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.act = LIFSpike(T=T)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        self.AP = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512 * block.expansion, num_classes)

        # Initialize weights kaiming_normal
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, MSBottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, MSBasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, self.T))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def change_activation(self, act):
        for m in self.modules():
            if isinstance(m, LIFSpike):
                m.set_ANN(act)

    def forward(self, x):
        out = self.bn1(self.conv1(x))
        # expand time
        if self.T > 1:
            out = self.expand_time(out)
        out = self.act(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.AP(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        if self.T != 1:
            out = out.view(self.T, -1, out.size(1))
            out = out.mean(0)
        return out

class MSResNet_SB(MSResNet):
    def __init__(self, block, num_blocks, sb_places, sb_pads=128, num_classes=10, T=1, zero_init_residual=False):
        super(MSResNet_SB, self).__init__(block, num_blocks=num_blocks, num_classes=num_classes,
                                          T=T, zero_init_residual=zero_init_residual)
        # put all the self.layers into a sequence
        layers = []
        channels = []
        for layeri in self.layer1:
            layers.append(layeri)
            channels.append(64)
        for layeri in self.layer2:
            layers.append(layeri)
            channels.append(128)
        for layeri in self.layer3:
            layers.append(layeri)
            channels.append(256)
        for layeri in self.layer4:
            layers.append(layeri)
            channels.append(512)
        self.layers = nn.Sequential(*layers)
        self.sb_places = []
        sb_sum = 0
        for i in range(len(sb_places)):
            sb_sum += sb_places[i]
            self.sb_places.append(sb_sum)
            if sb_sum > sum(num_blocks):
                raise ValueError("sb_places is not correct, sum of sb_places should be less than sum of num_blocks")
        self.sb_pads = sb_pads
        self.sb_layers = nn.ModuleList()
        sb_blocks = []
        for i in range(len(self.sb_places)):
            if channels[self.sb_places[i] - 1] == 64:
                kernels = [7, 5, 3]
            elif channels[self.sb_places[i] - 1] == 128:
                kernels = [7, 5, 3]
            elif channels[self.sb_places[i] - 1] == 256:
                kernels = [7, 5, 3]
            else:
                kernels = [7, 5, 3]
            sb_blocks.append(SurrogateBlock(kernels=kernels, in_channel=channels[self.sb_places[i]-1],
                                            out_channel=self.sb_pads, num_classes=num_classes, T=self.T))
        self.sb_layers = nn.ModuleList(sb_blocks)
        self.use_detach = False

        # initialize the weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, MSBottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, MSBasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def forward(self, x):
        outs = []
        out = self.bn1(self.conv1(x))
        # expand time
        if self.T > 1:
            out = self.expand_time(out)
        out = self.act(out)
        sb_i = 0
        for i in range(len(self.layers)):
            out = self.layers[i](out)
            if i+1 in self.sb_places:
                outs.append(self.sb_layers[sb_i](out))
                sb_i += 1
                if self.use_detach:
                    out = out.detach()

        out = self.AP(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        if self.T != 1:
            out = out.view(self.T, -1, out.size(1))
            out = out.mean(0)
        # put the out into the outs first place
        outs.insert(0, out)
        return outs



def MSResNet18(**kwargs):
    return MSResNet(MSBasicBlock, [2, 2, 2, 2], **kwargs)

def MSResNet34(**kwargs):
    return MSResNet(MSBasicBlock, [3, 4, 6, 3], **kwargs)

def MSResNet50(**kwargs):
    return MSResNet(MSBottleneck, [3, 4, 6, 3], **kwargs)

def MSResNet101(**kwargs):
    return MSResNet(MSBottleneck, [3, 4, 23, 3], **kwargs)

def MSResNet152(**kwargs):
    return MSResNet(MSBottleneck, [3, 8, 36, 3], **kwargs)

def MSResNet104(**kwargs):
    return MSResNet(MSBasicBlock, [3, 8, 32, 8], **kwargs)

def MSResNet_SB18(**kwargs):
    return MSResNet_SB(MSBasicBlock, [2, 2, 2, 2], **kwargs)

def MSResNet_SB104(**kwargs):
    return MSResNet_SB(MSBasicBlock, [3, 8, 32, 8], **kwargs)

# This paper uses a different convention for the ResNet.


if __name__ == '__main__':
    # net = MSResNet18()
    # y = net(torch.randn(1, 3, 32, 32))
    # print(y.size())
    # net = MSResNet50()
    # y = net(torch.randn(1, 3, 32, 32))
    # print(y.size())
    # net = MSResNet101()
    # y = net(torch.randn(1, 3, 32, 32))
    x = torch.rand(20, 3, 32, 32)
    net1 = MSResNet18(num_classes=10, T=2)
    state_dict = net1.state_dict()
    with torch.no_grad():
        y1 = net1(x)
    net = MSResNet_SB18(sb_places=[2,2,2], sb_pads=128, num_classes=10, T=2)# sb_places=[2,4,6],
    net.load_state_dict(state_dict, strict=False)
    with torch.no_grad():
        sbs = net(x)
    y2=sbs[0]
    print((y2-y1).abs().sum())

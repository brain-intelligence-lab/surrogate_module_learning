import torch
import torch.nn as nn
import torch.nn.functional as F
from models.ResBlock import BasicBlock, Bottleneck, RMSBasciblock, SingleBasicBlock
from models.spiking_layer import LIFSpike, ExpandTime, RateEncoding, singleLIF
from models.surrogate_block import SurrogateBlock,SpikeSurrogateBlock


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, T=1, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.T = T
        self.expand_time = ExpandTime(T=T)
        if num_classes <= 200:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        else:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.act = LIFSpike(T=T)
        if num_classes <= 200:
            self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        else:
            self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=2)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.AP = nn.AdaptiveAvgPool2d((1, 1))

        self.linear = nn.Linear(512 * block.expansion, num_classes)


        # initialize the weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # zero-initialize the residual blocks
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, T=self.T))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def change_activation(self, act):
        for m in self.modules():
            if isinstance(m, LIFSpike):
                m.set_ANN(act)

    def forward(self, x):
        # expand time
        if len(x.shape)==4:
            if self.T > 1:
                x = self.expand_time(x)
        else:
            # T, B, C, H, W -> B*T, C, H, W
            T = x.shape[0]
            if T != self.T:
                raise ValueError('T must be equal to {}'.format(self.T))
            x = x.reshape(-1, *x.shape[2:])
        out = self.bn1(self.conv1(x))
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

class ResNet_SB(ResNet):
    def __init__(self, block, num_blocks, sb_places, sb_kernels, sb_pads=128, num_classes=10, T=1, zero_init_residual=False):
        super(ResNet_SB, self).__init__(block, num_blocks=num_blocks, num_classes=num_classes,
                                          T=T, zero_init_residual=zero_init_residual)
        # put all the self.layers into a sequence
        self.layers = []
        channels = []
        self.static_input = True
        self.expand_time = ExpandTime(T=T)
        for layeri in self.layer1:
            self.layers.append(layeri)
            channels.append(64)
        for layeri in self.layer2:
            self.layers.append(layeri)
            channels.append(128)
        for layeri in self.layer3:
            self.layers.append(layeri)
            channels.append(256)
        for layeri in self.layer4:
            self.layers.append(layeri)
            channels.append(512)
        # self.layers = nn.Sequential(*layers)
        #################################################
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
            kerneli = sb_kernels[i]
            sb_blocks.append(SurrogateBlock(kernels=kerneli, in_channel=channels[self.sb_places[i] - 1],
                                            out_channel=self.sb_pads, num_classes=num_classes,
                                            static_input=self.static_input, T=self.T))
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
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def forward(self, x):
        outs = []
        # expand time
        # expand time
        if len(x.shape) == 4:
            if self.T > 1:
                x = self.expand_time(x)
        else:
            # T, B, C, H, W -> B*T, C, H, W
            T = x.shape[0]
            if T != self.T:
                raise ValueError('T must be equal to {}'.format(self.T))
            x = x.reshape(-1, *x.shape[2:])
        out = self.bn1(self.conv1(x))
        out = self.act(out)
        sb_i = 0
        for i in range(len(self.layers)):
            out = self.layers[i](out)
            if i + 1 in self.sb_places:
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

class ResNet_SB2(ResNet):
    def __init__(self, block, num_blocks, sb_places, kernels, sb_pads=128, num_classes=10, T=1, zero_init_residual=False):
        super(ResNet_SB2, self).__init__(block, num_blocks=num_blocks, num_classes=num_classes,
                                          T=T, zero_init_residual=zero_init_residual)
        # put all the self.layers into a sequence
        self.layers = []
        channels = []
        self.static_input = False
        self.expand_time = ExpandTime(T=T)
        for layeri in self.layer1:
            self.layers.append(layeri)
            channels.append(64)
        for layeri in self.layer2:
            self.layers.append(layeri)
            channels.append(128)
        for layeri in self.layer3:
            self.layers.append(layeri)
            channels.append(256)
        for layeri in self.layer4:
            self.layers.append(layeri)
            channels.append(512)
        # self.layers = nn.Sequential(*layers)
        self.sb_places = []
        sb_sum = 0
        for i in range(len(sb_places)):
            sb_sum += sb_places[i]
            self.sb_places.append(sb_sum)
            if sb_sum > sum(num_blocks):
                raise ValueError("sb_places is not correct, sum of sb_places should be less than sum of num_blocks")
        self.sb_pads = sb_pads
        set_channels = []
        for i in range(len(sb_places)):
            set_channels.append(channels[self.sb_places[i] - 1])
        set_channels.append(channels[-1])
        self.sb_layers = SpikeContinueSB(kernels=kernels, channels=set_channels,
                                         num_classes=num_classes, static_input=self.static_input, T=self.T)

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
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def forward(self, x):
        puts = []
        # expand time
        # expand time
        if len(x.shape) == 4:
            if self.T > 1:
                x = self.expand_time(x)
        else:
            # T, B, C, H, W -> B*T, C, H, W
            T = x.shape[0]
            if T != self.T:
                raise ValueError('T must be equal to {}'.format(self.T))
            x = x.reshape(-1, *x.shape[2:])
        out = self.bn1(self.conv1(x))
        out = self.act(out)
        sb_i = 0
        for i in range(len(self.layers)):
            out = self.layers[i](out)
            if i + 1 in self.sb_places:
                puts.append(out)
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
        sb_out = self.sb_layers(puts)
        outs = [out, sb_out]
        return outs


class Small_Network(nn.Module):
    def __init__(self, num_classes=10, T=1):
        super(Small_Network, self).__init__()
        self.T = T
        self.static_input = True
        self.expand_time = ExpandTime(T=T)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.act = LIFSpike(T=self.T)
        self.layer1 = BasicBlock(in_planes=64, planes=64, stride=1, T=self.T)
        self.SB1 = SpikeSurrogateBlock(kernels=[7,5,3], in_channel=64,
                                            out_channel=256, num_classes=num_classes,
                                            static_input=self.static_input, T=self.T)
        self.use_detach = True

        # initialize the weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def change_activation(self, act):
        for m in self.modules():
            if isinstance(m, LIFSpike):
                m.set_ANN(act)

    def forward(self, x):
        # expand time
        if self.static_input and self.T > 1:
            x = self.expand_time(x)
        out = self.bn1(self.conv1(x))
        out = self.act(out)
        out = self.layer1(out)
        out = self.SB1(out)
        return out


def ResNet18(**kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)

def RResNet18(**kwargs):
    return ResNet(RBasicBlock, [2, 2, 2, 2], **kwargs)

def RMSResNet18(**kwargs):
    model = ResNet(RMSBasciblock, [2, 2, 2, 2], **kwargs)
    model.act = nn.Sequential()
    model.AP = nn.Sequential(
        LIFSpike(model.T),
        nn.AdaptiveAvgPool2d(1),
    )
    return model

def ResNet17(**kwargs):
    return ResNet(Bottleneck, [1, 1, 1, 2], **kwargs)

def ResNet34(**kwargs):
    return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)

def ResNet50(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)

def ResNet101(**kwargs):
    return ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)

def ResNet152(**kwargs):
    return ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)

def ResNet104(**kwargs):
    return ResNet(BasicBlock, [3, 8, 32, 8], **kwargs)

def ResNet_SB18(**kwargs):
    return ResNet_SB(BasicBlock, [2, 2, 2, 2], **kwargs)

def ResNet_SB34(**kwargs):
    return ResNet_SB(BasicBlock, [3, 4, 6, 3], **kwargs)

def ResNet_SB104(**kwargs):
    return ResNet_SB(BasicBlock, [3, 8, 32, 8], **kwargs)

def ResNet_SB218(**kwargs):
    return ResNet_SB2(BasicBlock, [2, 2, 2, 2], **kwargs)


if __name__ == '__main__':
    x = torch.rand(2, 3, 32, 32)
    y = torch.rand(2, 10)
    sb_poi = [2, 2, 2]
    kernels = [[7, 5, 3], [7, 5, 3], [7, 5, 3]]
    sb_pads = 256
    model = ResNet_SB18(sb_kernels=kernels, sb_places=sb_poi, sb_pads=sb_pads, T=1)
    # model = ResNet18(T=1)
    print(model)
    out = model(x)
    print(out[0].shape)

    loss = F.mse_loss(out[0], y)+F.mse_loss(out[1], y)+F.mse_loss(out[2], y)+F.mse_loss(out[3], y)
    loss.backward()
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            if m.weight.grad is None:
                print(m)



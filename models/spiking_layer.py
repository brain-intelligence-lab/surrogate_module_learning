import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class ZIF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, gama):
        out = (input >= 0).float()
        L = torch.tensor([gama])
        ctx.save_for_backward(input, L)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (input, others) = ctx.saved_tensors
        gama = others[0].item()

        grad_input = grad_output.clone()
        tmp = ((1 / gama) * (1 / gama) * ((gama - input.abs())).clamp(min=0))
        grad_input = grad_input * tmp
        return grad_input, None


class DSPIKE(nn.Module):
    def __init__(self, region=1.0):
        super(DSPIKE, self).__init__()
        self.region = region

    def forward(self, x, temp):
        out_bpn = torch.clamp(x, -self.region, self.region)
        out_bp = (torch.tanh(temp * out_bpn)) / \
                 (2 * np.tanh(self.region * temp)) + 0.5
        out_s = (x >= 0).float()
        return (out_s.float() - out_bp).detach() + out_bp

class BDspike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, gamma):
        out = (input >= 0).float()
        L = torch.tensor([gamma])
        ctx.save_for_backward(input, L)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (input, others) = ctx.saved_tensors
        gama = others[0].item()
        gamma1 = 1.0
        gamma2 = 2.5
        r = 2.0
        grad_input = grad_output.clone()
        # tmp1 = ((1 / gamma1) * (1 / gamma1) * ((gamma1 - input.abs())).clamp(min=0))
        # tmp2 = ((1 / gamma2) * (1 / gamma2) * ((gamma2 - input.abs())).clamp(min=0))
        # sigma = 1/4*(tmp1**2 - tmp2**2)
        # tmp = (tmp1 * tmp2 - sigma)**0.5
        out_bpn = torch.clamp(input, -r, r)
        k1 = 1/(2 * np.tanh(r * gamma1))
        k2 = 1/(2 * np.tanh(r * gamma2))
        tmp1 = gamma1 * k1 * (1 - torch.tanh(gamma1 * out_bpn)**2)
        tmp2 = gamma2 * k2 * (1 - torch.tanh(gamma2 * out_bpn)**2)
        # sigma = 0.25 * ((tmp1+tmp2) ** 2 - (tmp1-tmp2) ** 2)+1e-8
        tmp = tmp2 * 1.2 - tmp1 * 0.2
        # tmp = (tmp1 + tmp2) / 2
        # tmp = tmp2
        grad_input = grad_input * tmp
        return grad_input, None

class LIFSpike(nn.Module):
    def __init__(self, T=1, thresh=1.0, tau=0.5, gamma=2.5, use_ann=False):
        super(LIFSpike, self).__init__()
        self.use_ann = use_ann
        self.act_ann = nn.ReLU()
        self.snn_act = DSPIKE(region=1.0)
        self.T = T # time steps
        self.thresh = thresh
        self.tau = tau
        self.gamma = gamma
        self.mem_detach = False

    def forward(self, x):
        if self.use_ann:
            return self.act_ann(x)
        else:
            if self.T == 1:
                spike =  self.snn_act(x - self.thresh, self.gamma)
                # TODO: maybe compute the KL divergence between the two distributions
                return spike
            else:
                # adjust the size of x (B*T,C,H,W) to (B,T,C,H,W)
                x = x.view(self.T, -1, *x.shape[1:])
                mem = 0
                spikes = torch.zeros_like(x)
                for t in range(self.T):
                    # calculate the membrane potential
                    mem = mem * self.tau + x[t,...]
                    spike = self.snn_act(mem - self.thresh, self.gamma)
                    spikes[t,...] = spike
                    if self.mem_detach:
                        mem = mem * (1 - spike.detach())
                    else:
                        mem = mem * (1 - spike)
                # adjust the size of spikes (B,T,C,H,W) to (B*T,C,H,W)
                return spikes.view(-1, *spikes.shape[2:])

    def set_ANN(self, act: bool):
        self.use_ann = act

    def set_mem_detach(self, mem_detach):
        self.mem_detach = mem_detach

class singleLIF(nn.Module):
    def __init__(self, thresh=1.0, tau=0.5, gamma=2.5):
        super(singleLIF, self).__init__()
        self.act = DSPIKE(region=1.0)
        self.thresh = thresh
        self.tau = tau
        self.gamma = gamma
        self.mem = 0

    def forward(self, x):
        self.mem = self.mem * self.tau + x
        spike = self.act(self.mem - self.thresh, self.gamma)
        self.mem = self.mem * (1 - spike)
        return spike

    def reset(self):
        self.mem = 0


class LIAFSpike(nn.Module):
    def __init__(self, T=1, thresh=1.0, tau=0.5):
        super(LIAFSpike, self).__init__()
        self.use_ann = False
        self.act_ann = nn.ReLU()
        # self.snn_act = BDspike.apply
        self.T = T # time steps
        self.thresh = thresh
        self.tau = tau
        self.mem_detach = False

    def forward(self, x):
        if self.T == 1:
            return self.act_ann(x - self.thresh)
        else:
            # adjust the size of x (B*T,C,H,W) to (B,T,C,H,W)
            x = x.view(self.T, -1, *x.shape[1:])
            mem = 0
            outs = torch.zeros_like(x)
            for t in range(self.T):
                # calculate the membrane potential
                mem = mem * self.tau + x[t,...]
                out = self.act_ann(mem - self.thresh)
                spike = (mem >= self.thresh).float().detach()
                outs[t,...] = out
                mem = mem * (1 - spike)
            # adjust the size of spikes (B,T,C,H,W) to (B*T,C,H,W)
            return outs.view(-1, *outs.shape[2:])

    def set_ANN(self, act: bool):
        self.use_ann = act

    def set_mem_detach(self, mem_detach):
        self.mem_detach = mem_detach

class ExpandTime(nn.Module):
    def __init__(self, T=1):
        super(ExpandTime, self).__init__()
        self.T = T

    def forward(self, x):
        x_seq = x[None,:,:,:,:]
        x_seq = x_seq.repeat(self.T, 1 , 1, 1, 1)
        # adjust the size of spikes (B,T,C,H,W) to (B*T,C,H,W)
        return x_seq.view(-1, *x_seq.shape[2:])

class RateEncoding(nn.Module):
    def __init__(self, T=1):
        super(RateEncoding, self).__init__()
        self.T = T

    def forward(self, x):
        x_seq = x[None, :, :, :, :]
        x_seq = x_seq.repeat(self.T, 1, 1, 1, 1)
        x_seq = x_seq.view(-1, *x_seq.shape[2:])
        # poision noise
        noise = torch.randn_like(x_seq)
        x_seq = (x_seq>=noise).float().detach()
        return x_seq

if __name__ == '__main__':
    x = torch.randn(2,3,8,8)
    expt = ExpandTime(T=2)
    y = expt(x)
    print(y.shape)



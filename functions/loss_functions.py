import torch
import torch.nn as nn
import torch.nn.functional as F


def tet_loss(outputs, labels, criterion, means, lamb):
    T = outputs.size(0)
    Loss_es = 0
    for t in range(T):
        Loss_es += criterion(outputs[t, ...], labels)
    Loss_es = Loss_es / T
    if lamb != 0:
        MMDLoss = torch.nn.MSELoss()
        y = torch.zeros_like(outputs).fill_(means)
        Loss_mmd = MMDLoss(outputs, y)
    else:
        Loss_mmd = 0
    return (1 - lamb) * Loss_es + lamb * Loss_mmd


class TET_loss_with_MMD(nn.Module):
    def __init__(self, criterion, lamb=0.01, means=1.0):
        super(TET_loss_with_MMD, self).__init__()
        self.criterion = criterion
        self.lamb = lamb
        self.means = means

    def forward(self, outputs, labels):
        return tet_loss(outputs, labels, self.criterion, self.means, self.lamb)


def mse_loss(outputs, labels):
    MMDLoss = torch.nn.MSELoss()
    out = outputs.mean(1)
    target_m = F.one_hot(labels, out.size(-1)).float()
    loss = MMDLoss(out, target_m)
    return loss


def SoftCrossEntropy(inputs, target, temperature):
    log_likelihood = -F.log_softmax(inputs / temperature, dim=1)
    batch = inputs.shape[0]
    loss = torch.sum(torch.mul(log_likelihood, F.softmax(target.detach() / temperature, dim=1))) / batch
    return loss

def simple_loss(p, z):
    p = nn.functional.normalize(p, dim=1)
    z = nn.functional.normalize(z, dim=1)
    return -(p * z).sum(dim=1).mean()

def SoftKL(inputs, target, temperature):
    criterion = nn.KLDivLoss(reduction = 'batchmean')
    log_likelihood = F.log_softmax(inputs / temperature, dim=1)
    # batch = inputs.shape[0]
    target_pro = F.softmax(target.detach() / temperature, dim=1)
    loss = criterion(log_likelihood, target_pro)
    return loss

class SBDistillationLoss(nn.Module):
    def __init__(self, criterion, temperature=3.0, alpha=1.0, lamb=0.333):
        super(SBDistillationLoss, self).__init__()
        self.criterion = criterion
        self.temperature = temperature
        self.lamb = lamb  # the weight of the soft loss
        self.alpha = alpha  # the weight of the auxiliary output hard loss
        self.special_loss = criterion# nn.CrossEntropyLoss(label_smoothing=0.2)

    def forward(self, inputs, target):
        # inputs is a list of network output [final output, intermediate outputs]
        finput = inputs[0]
        hard_loss = self.criterion(finput, target)
        soft_loss = 0  # soft loss is the kl divergence loss to make the output of the final output and the intermediate outputs close to each other
        for i in range(1, len(inputs)):
            hard_loss += self.special_loss(inputs[i], target) * self.alpha
        hard_loss = hard_loss / ((len(inputs) - 1) * self.alpha + 1)
        if self.lamb > 0:
            for i in range(1, len(inputs)):
                soft_loss += SoftKL(inputs[i], finput, self.temperature)  # TODO: check the temperature
                soft_loss += SoftKL(finput, inputs[i],self.temperature)  # making the final output better # normalize the hard loss

            soft_loss = soft_loss / (2 * (len(inputs) - 1)+1e-8)  # normalize the soft loss
        return self.lamb * soft_loss + (1 - self.lamb) * hard_loss
        # return self.lamb * soft_loss +  hard_loss

class SBDistillationLoss2(nn.Module):
    def __init__(self, criterion, temperature=3.0, alpha=1.0, lamb=0.333):
        super(SBDistillationLoss2, self).__init__()
        self.criterion = criterion
        self.temperature = temperature
        self.lamb = lamb  # the weight of the soft loss
        self.alpha = alpha  # the weight of the auxiliary output hard loss
        self.special_loss = criterion# nn.CrossEntropyLoss(label_smoothing=0.2)

    def forward(self, inputs, target):
        # inputs is a list of network output [final output, intermediate outputs]
        finput = inputs[0]
        hard_loss = self.criterion(finput, target)
        soft_loss = 0  # soft loss is the kl divergence loss to make the output of the final output and the intermediate outputs close to each other
        for i in range(1, len(inputs)):
            hard_loss += self.special_loss(inputs[i], target) * self.alpha
        hard_loss = hard_loss / ((len(inputs) - 1) * self.alpha + 1)
        if self.lamb > 0:
            for i in range(1, len(inputs)):
                soft_loss += SoftKL(inputs[i], finput, self.temperature)  # TODO: check the temperature
                soft_loss += SoftKL(finput, inputs[i],self.temperature)  # making the final output better # normalize the hard loss

            soft_loss = soft_loss / (1 * (len(inputs) - 1)+1e-8)  # normalize the soft loss
        return self.lamb * soft_loss + (1 - self.lamb) * hard_loss
        # return self.lamb * soft_loss +  hard_loss


class SimCLRLoss(nn.Module):
    def __init__(self, T,criterion):
        super(SimCLRLoss, self).__init__()
        self.T = T
        self.temperature = 0.5
        self.criterion = criterion

    def sim_loss(self, out1, out2):
        # simclr loss
        batch_size = out1.size(0)
        out = torch.cat([out1, out2], dim=0)
        sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / self.temperature)
        mask = (torch.ones_like(sim_matrix) - torch.eye(2 * batch_size, device=sim_matrix.device)).bool()
        # [2*B, 2*B-1]
        sim_matrix = sim_matrix.masked_select(mask).view(2 * batch_size, -1)
        # compute loss
        pos_sim = torch.exp(torch.sum(out1 * out2, dim=-1) / self.temperature)
        # [2*B]
        pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
        return (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()


    def time_sim_loss(self, inputs):
        # reconstruction time dimention
        inputs = inputs.view(-1, self.T, inputs.size(-1))
        loss = 0
        for i in range(self.T):
            for j in range(i + 1, self.T):
                loss += self.sim_loss(inputs[:, i, :], inputs[:, j, :])
        return loss / (self.T * (self.T - 1) / 2)

    def forward(self, inputs, target):
        hard_loss = self.criterion(inputs[0], target)  # hard loss is the classification loss
        sim_loss = 0
        for i in range(1, len(inputs)):
            sim_loss += self.time_sim_loss(inputs[i])
        return hard_loss + sim_loss


if __name__ == '__main__':
    x = torch.randn(2, 10, 10)
    # x.requires_grad = True
    t = torch.randn(2, 10, 10)
    loss1 = SoftCrossEntropy(x, t, 1.0)
    loss2 = SoftKL(x, t, 1.0)
    print(loss1)
    print(loss2)
    # loss2.backward()
    # print(x.grad)
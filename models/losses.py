
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function, Variable

def cross_entropy_2d(input, target, weight=None, size_average=True):
    n, c, h, w = input.size()
    log_p = F.log_softmax(input, dim=1)
    log_p = log_p.transpose(1,2).transpose(2,3).contiguous().view(-1,c)
    target = target.view(target.numel())
    loss = F.nll_loss(log_p, target, weight=weight, size_average=False)
    if size_average:
        loss /= float(target.numel())
    return loss

def cross_entropy_3d(input, target, weight=None, size_average=True):
    n, c, h, w, s = input.size()
    log_p = F.log_softmax(input, dim=1)
    log_p = log_p.transpose(1,2).transpose(2,3).transpose(3,4).contiguous().view(-1,c)
    target = target.view(target.numel())
    loss = F.nll_loss(log_p, target, weight=weight, size_average=False)
    if size_average:
        loss/= float(target.numel())
    return loss

class SoftDiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(SoftDiceLoss, self).__init__()
        self.one_hot_encoder = OneHot(n_classes).forward
        self.n_classes = n_classes

    def forward(self, input, target):
        smooth = 0.01
        batch_size = input.size(0)

        input = F.softmax(input, dim=1).view(batch_size, self.n_classes, -1)
        target = self.one_hot_encoder(target).contiguous().view(batch_size, self.n_classes, -1)

        inter = torch.sum(input*target,2) + smooth
        union = torch.sum(input,2)+torch.sum(target,2)+smooth

        score = torch.sum(2.0*inter/union)
        score = 1.0 - score / (float(batch_size)*float(self.n_classes))

        return score

class OneHot(nn.Module):
    def __init__(self, depth):
        super(OneHot, self).__init__()
        self.depth = depth
        # self.ones = torch.sparse.torch.eye(depth).cuda()
        self.ones = torch.sparse.torch.eye(depth)

    def forward(self, x_in):
        n_dim = x_in.dim()
        output_size = x_in.size() + torch.Size([self.depth])
        num_element = x_in.numel()
        x_in = x_in.data.long().view(num_element)
        out = Variable(self.ones.index_select(0,x_in)).view(output_size)
        return out.permute(0, -1, *range(1, n_dim)).squeeze(dim=2).float()

    def __repr__(self):
        return self.__class__.__name__ + "({})".format(self.depth)
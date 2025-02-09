import torch
import torch.nn as nn

from torch.autograd import Variable
from . import base

import segmentation_models_pytorch as smp

class CombinedLoss(base.Loss):
    
    def __init__(self, losses, weights, **kwargs):
        super().__init__(**kwargs)
        self.losses = losses
        if len(self.losses) == 1:
            self.weights = [1.0]
        else:
            self.weights = weights
        
    def forward(self, x_embeddings, y_embeddings):
        
        if len(self.losses) != len(self.weights):
            raise ValueError("O número de funções de loss deve ser igual ao número de pesos.")
    
        combined_loss = Variable(torch.Tensor([0])).cuda()
        
        for i in range(len(self.losses)):
            combined_loss += self.weights[i] * self.losses[i](x_embeddings, y_embeddings)
            
        return combined_loss

class FocalLoss(base.Loss):
    
    def __init__(self, ignore_index=255, **kwargs):
        super().__init__(**kwargs)
        self.focal = smp.losses.FocalLoss(gamma=2, mode='multiclass', ignore_index=ignore_index)
        
    def forward(self, x_embeddings, y_embeddings):
        focal_loss = self.focal(x_embeddings, y_embeddings)
        return focal_loss

class CrossEntropyLoss(nn.CrossEntropyLoss, base.Loss):
    pass

# LMCE
# https://github.com/CoinCheung/pytorch-loss/blob/master/large_margin_softmax.py
class LargeMarginInSoftmaxLoss(base.Loss):

    def __init__(self, lam=0.3, reduction='mean', ignore_index=255):
        super(LargeMarginInSoftmaxLoss, self).__init__()
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.lam = lam
        self.logit_norm = True
        self.t = 0.02

    def forward(self, logits, labels):
        logits = logits.float()
        mask = labels == self.ignore_index
        lb = labels.clone().detach()
        lb[mask] = 0
        loss = LargeMarginSoftmaxFuncV2.apply(logits, lb, self.lam)
        loss = loss[mask == 0]
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss

class LargeMarginSoftmaxFuncV2(torch.autograd.Function):

    @staticmethod
    def forward(ctx, logits, labels, lam=0.3, focal_flag=False):
        num_classes = logits.size(1)
        coeff = 1. / (num_classes - 1.)
        idx = torch.zeros_like(logits).scatter_(1, labels.unsqueeze(1), 1.)

        lgts = logits.clone()
        lgts[idx.bool()] = -1.e6
        q = lgts.softmax(dim=1)
        log_q = lgts.log_softmax(dim=1)
        losses = q.sub_(coeff).mul_(log_q).mul_(lam / 2.)
        losses[idx.bool()] = 0

        losses = losses.sum(dim=1).add_(nn.functional.cross_entropy(logits, labels, reduction='none', ignore_index=255))
        
        ctx.variables = logits, labels, idx, coeff, lam
        return losses

    @staticmethod
    def backward(ctx, grad_output):
        '''
        compute gradient
        '''
        logits, labels, idx, coeff, lam = ctx.variables
        num_classes = logits.size(1)

        p = logits.softmax(dim=1)
        lgts = logits.clone()
        lgts[idx.bool()] = -1.e6
        q = lgts.softmax(dim=1)
        qx = q * lgts
        qx[idx.bool()] = 0

        grad = qx + q - q * qx.sum(dim=1).unsqueeze(1) - coeff
        grad = grad * lam / 2.
        grad[idx.bool()] = -1
        grad = grad + p

        grad.mul_(grad_output.unsqueeze(1))

        return grad, None, None

# LMFCL
class LargeMarginFocalLoss(base.Loss):

    def __init__(self, lam=0.3, reduction='mean', ignore_index=255):
        super(LargeMarginFocalLoss, self).__init__()
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.lam = lam
        self.focal_crit = FocalLoss(ignore_index=255)

    def forward(self, logits, label):
        # overcome ignored label
        logits = logits.float()
        with torch.no_grad():
            num_classes = logits.size(1)
            coeff = 1. / (num_classes - 1.)
            lb = label.clone().detach()
            mask = label == self.ignore_index
            lb[mask] = 0
            idx = torch.zeros_like(logits).scatter_(1, lb.unsqueeze(1), 1.)

        # large margin in softmax loss
        lgts = logits - idx * 1.e6
        q = lgts.softmax(dim=1)
        q = q * (1. - idx)

        log_q = lgts.log_softmax(dim=1)
        log_q = log_q * (1. - idx)
        mg_loss = ((q - coeff) * log_q) * (self.lam / 2)
        mg_loss = mg_loss * (1. - idx)
        mg_loss = mg_loss.sum(dim=1)

        ce_loss = self.focal_crit(logits, label)
        # sum general focal loss + large magin of each pixel
        loss = ce_loss + mg_loss
        loss = loss[mask == 0]

        if self.reduction == 'mean':
            loss = loss.mean()
        if self.reduction == 'sum':
            loss = loss.sum()

        return loss
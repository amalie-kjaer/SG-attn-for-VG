import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, y):
        ce_loss = F.cross_entropy(pred, y, reduction='none')
        pt = torch.exp(-ce_loss)
        loss = (self.alpha[y] * (1 - pt) ** self.gamma * ce_loss).mean()
        return loss
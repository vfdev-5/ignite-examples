import torch
from torch.nn import Module
import torch.nn.functional as F


class FocalLoss(Module):
    def __init__(self, gamma=1.0, reduce=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduce = reduce

    def forward(self, input, target):
        xentropy = F.cross_entropy(input, target, reduce=False)
        target = target.view(-1, 1)
        w = (1.0 - F.softmax(input, dim=1).gather(1, target)).view(-1)
        w = torch.pow(w, self.gamma)
        if self.reduce:
            return torch.mean(w * xentropy)
        return w * xentropy

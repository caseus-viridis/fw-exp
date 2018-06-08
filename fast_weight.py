import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def fast_weight_update(A, x1, x2, lam=1., eta=1.):
    return A * lam + torch.bmm(x1.unsqueeze(2), x2.unsqueeze(1)) * eta

class FastWeight(nn.Module):
    """
    Fast weight
    """
    def __init__(self, dim, lam=1., eta=1.):
        super(FastWeight, self).__init__()
        self.lam = lam
        self.eta = eta

    def forward(self, A, h):
        return fast_weight_update(A, h, h, self.lam, self.eta)

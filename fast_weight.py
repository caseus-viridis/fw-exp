import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def fast_weight_update(A, x1, x2, lam=1., eta=1.):
    return A * lam + torch.bmm(x1.unsqueeze(2), x2.unsqueeze(1)) * eta

def fast_weight_query(A, x):
    return torch.matmul(A, x.unsqueeze(2)).squeeze()

class FastWeight(nn.Module):
    """
    Fast weight
    """
    def __init__(self, dim, lam=1., eta=1.):
        super(FastWeight, self).__init__()
        self.lam = lam
        self.eta = eta

    def forward(self, A, x):
        A_ = fast_weight_update(A, x, x, self.lam, self.eta)
        x_ = fast_weight_query(A_, x)
        return (A_, x_)

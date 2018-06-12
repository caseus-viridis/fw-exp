import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def fast_weight_update(A, x1, x2, lam=1., eta=1., mode='hebb'):
    x1_x2_outer = torch.bmm(x1.unsqueeze(2), x2.unsqueeze(1))
    if mode == 'hebb':
        return lam * A + eta * x1_x2_outer
    elif mode == 'oja':
        x1_x2_inner = torch.bmm(x1.unsqueeze(1), x2.unsqueeze(2))
        return (1. - eta * x1_x2_inner) * A + eta * x1_x2_outer
    elif mode == 'sanger':
        return A #[TODO] implement this
    else:
        raise RuntimeError("Unknown fast weight update mode: {}".format(mode))


def fast_weight_query(A, x):
    return torch.matmul(A, x.unsqueeze(2)).squeeze()


class FastWeight(nn.Module):
    """
    Fast weight
    """

    def __init__(self, hidden_size, lam=1., eta=1., mode='hebb'):
        super(FastWeight, self).__init__()
        self.hidden_size = hidden_size
        self.lam = lam
        self.eta = eta
        self.mode = mode

    def read(self, A, x):
        return fast_weight_query(A, x)

    def write(self, A, x1, x2):
        return fast_weight_update(A, x1, x2, self.lam, self.eta, self.mode)

    def forward(self, A, x):
        A_ = self.write(A, x, x)
        x_ = self.read(A_, x)
        return (A_, x_)

    def extra_repr(self):
        return "hidden_size = {}, lambda = {}, eta = {}, mode = {}".format(
            self.hidden_size, self.lam, self.eta, self.mode)

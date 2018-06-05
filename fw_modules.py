import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

def fw_update(A, h, lam=1., eta=1.):
    return A


class FastWeight(nn.Module):
    """
    Fast weight
    """
    def __init__(self, dim, lam=1., eta=1.):
        super(FastWeight, self).__init__()
        self.lam = lam
        self.eta = eta
        self.register_buffer('A', None)

    def forward(self, querry):
        return self.A * 


class FastWeightRNNCellBase(nn.RNNCellBase):

    def check_forward_hidden(self, input, hx, A, hidden_label=''):
        super(FastWeightRNNCellBase).check_forward_hidden(self, input, hx, hidden_label)
        if input.size(0) != A.size(0):
            raise RuntimeError(
                "Input batch size {} doesn't match fast-weight{} batch size {}".format(
                    input.size(0), hidden_label, A.size(0)))
        if A.size(1) != self.hidden_size:
            raise RuntimeError(
                "fast-weight{} has inconsistent hidden_size: got {}, expected {}".format(
                    hidden_label, A.size(1), self.hidden_size))


class FWRNNCell(FastWeightRNNCellBase):
    """
    Fast weight RNN cell
    """
    def __init__(self, input_size, hidden_size, bias=True, nonlinearity="relu"):
        super(FWRNNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.nonlinearity = nonlinearity

        self.rec_slow = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.inp_slow = nn.Linear(hidden_size, hidden_size, bias=bias)
        if self.nonlinearity == "tanh":
            self.act = nn.Tanh()
        elif self.nonlinearity == "relu":
            self.act = nn.ReLU()
        else:
            raise RuntimeError(
                "Unknown nonlinearity: {}".format(self.nonlinearity))

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for p in self.parameters():
            p.data.uniform_(-stdv, stdv)

    def forward(self, input, hx, A):
        self.check_forward_input(input)
        self.check_forward_hidden(input, hx, A)
        self.rec_slow(hx) +

        return func(
            input, hx,
            self.weight_ih, self.weight_hh,
            self.bias_ih, self.bias_hh,
        )

class FWLSTMCell(FastWeightRNNCellBase):
    pass

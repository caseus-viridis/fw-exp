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


class FastWeightRNNCellBase(nn.Module):

    def extra_repr(self):
        s = '{input_size}, {hidden_size}'
        if 'bias' in self.__dict__ and self.bias is not True:
            s += ', bias={bias}'
        if 'nonlinearity' in self.__dict__ and self.nonlinearity != "tanh":
            s += ', nonlinearity={nonlinearity}'
        return s.format(**self.__dict__)

    def check_forward_input(self, input):
        if input.size(1) != self.input_size:
            raise RuntimeError(
                "input has inconsistent input_size: got {}, expected {}".format(
                    input.size(1), self.input_size))

    def check_forward_hidden(self, input, hx, A, hidden_label=''):
        if input.size(0) != hx.size(0):
            raise RuntimeError(
                "Input batch size {} doesn't match hidden{} batch size {}".format(
                    input.size(0), hidden_label, hx.size(0)))
        if hx.size(1) != self.hidden_size:
            raise RuntimeError(
                "hidden{} has inconsistent hidden_size: got {}, expected {}".format(
                    hidden_label, hx.size(1), self.hidden_size))
        if input.size(0) != A.size(0):
            raise RuntimeError(
                "Input batch size {} doesn't match fast-weight{} batch size {}".format(
                    input.size(0), hidden_label, A.size(0)))
        if A.size(1) != self.hidden_size:
            raise RuntimeError(
                "fast-weight{} has inconsistent hidden_size: got {}, expected {}".format(
                    hidden_label, A.size(1), self.hidden_size))


class FastWeightRNNCell(FastWeightRNNCellBase):
    """
    Fast weight RNN cell, with layer norm
    Ba et al. 2016
    """
    def __init__(self, input_size, hidden_size, bias=True, nonlinearity="relu", lam=1., eta=1.):
        super(FastWeightRNNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.nonlinearity = nonlinearity
        self.lam = lam
        self.eta = eta

        self.rec_slow = nn.Linear(hidden_size, hidden_size, bias=False)
        self.inp_slow = nn.Linear(input_size, hidden_size, bias=bias)
        self.fast = FastWeight(hidden_size, lam=lam, eta=eta)
        self.ln = nn.LayerNorm(hidden_size)

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

    def forward(self, x, h, A):
        self.check_forward_input(x)
        self.check_forward_hidden(x, h, A)
        h_ = self.act(self.ln(self.rec_slow(h) + self.inp_slow(x)))
        A_ = self.fast(A, h_)
        return h_, A_


class FastWeightLSTMCell(FastWeightRNNCellBase):
    """
    Fast weight LSTM cell
    Keller et al. 2018
    """
    pass


class RNNCellStack(nn.Module):
    """
    Multiplayer RNN cells
    """
    def __init__(self, cell_list):
        self.cell_list = cell_list
        self.num_layers = len(self.cell_list)
        check_sizes(self.cell_list)

    @staticmethod
    def check_sizes(cell_list):
        hidden_size = None
        for cell in cell_list:
            if hidden_size is not None and cell.input_size!=hidden_size:
                raise RuntimeError(
                    "cell {} has input_size {}, not matching previous hidden_size {}".format(
                        cell, cell.input_size, hidden_size))
            hidden_size = cell.hidden_size

    def forward(self, input, state):
        for cell in self.cell_list:
            cell(input, state)


class FastWeightRNN(nn.Module):
    """
    Fast weight RNN
    """
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, nonlinearity="relu", lam=1., eta=1.):
        super(FastWeightRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.nonlinearity = nonlinearity
        self.lam = lam
        self.eta = eta
        # self.h0 = None
        # self.h = []

        self.cells = [FastWeightRNNCell(self.input_size if _i==0 else self.hidden_size, self.hidden_size, self.bias, self.nonlinearity, self.lam, self.eta) for _i in range(self.num_layers)]

    def init_states(self):
        for h_ in self.h:
            h_.fill_(0.)
        self.A.fill_(0.)

    def forward(self, input, h, A):
        batch_size = input.shape(0)
        time_steps = input.shape(1)
        self.cell(input, self.state)

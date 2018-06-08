import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from fast_weight import FastWeight


class _RNNCellBase(nn.Module):

    def extra_repr(self):
        s = 'input_size={input_size}, hidden_size={hidden_size}'
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

    def check_forward_hidden(self, input, hx, hidden_label=''):
        if input.size(0) != hx.size(0):
            raise RuntimeError(
                "Input batch size {} doesn't match hidden{} batch size {}".format(
                    input.size(0), hidden_label, hx.size(0)))
        if hx.size(1) != self.hidden_size:
            raise RuntimeError(
                "hidden{} has inconsistent hidden_size: got {}, expected {}".format(
                    hidden_label, hx.size(1), self.hidden_size))

    def check_forward_fast_weight(self, input, hx, A, hidden_label=''):
        if input.size(0) != A.size(0):
            raise RuntimeError(
                "Input batch size {} doesn't match fast-weight{} batch size {}".format(
                    input.size(0), hidden_label, A.size(0)))
        if A.size(1) != self.hidden_size:
            raise RuntimeError(
                "fast-weight{} has inconsistent hidden_size: got {}, expected {}".format(
                    hidden_label, A.size(1), self.hidden_size))

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for p in self.parameters():
            p.data.uniform_(-stdv, stdv)

    def get_nonlinearity(self):
        if self.nonlinearity == "tanh":
            return nn.Tanh()
        elif self.nonlinearity == "relu":
            return nn.ReLU()
        else:
            raise RuntimeError(
                "Unknown nonlinearity: {}".format(self.nonlinearity))

    def register_state_buffer(self):
        raise NotImplementedError("RNN cell instance {} does not have register_state_buffer() implemented".format(self))

    def state(self):
        raise NotImplementedError("RNN cell instance {} does not have state() implemented".format(self))


class RNNCellStack(nn.Module):
    """
    Multiplayer RNN cells
    """
    def __init__(self, cell_list):
        super(RNNCellStack, self).__init__()
        self.cell_list = cell_list
        self.num_layers = len(self.cell_list)
        self.check_sizes(self.cell_list)
        for i, layer in enumerate(self.cell_list):
            self.add_module(str(i), layer)

    def __len__(self):
        return self.num_layers

    def __getitem__(self, idx):
        assert idx < self.__len__(), "layer index {} out of bound ({})".format(idx, self.__len__())
        return self.cell_list[idx]

    @staticmethod
    def check_sizes(cell_list):
        hidden_size = None
        for cell in cell_list:
            if hidden_size is not None and cell.input_size!=hidden_size:
                raise RuntimeError(
                    "cell {} has input_size {}, not matching previous hidden_size {}".format(
                        cell, cell.input_size, hidden_size))
            hidden_size = cell.hidden_size

    def forward(self, input, states):
        output = input
        for cell, state in zip(self.cell_list, states):
            output, state = cell(output, state)
        return output, states

    def register_state_buffers(self, batch_size):
        for cell in self.cell_list:
            cell.register_state_buffer(batch_size)

    def states(self):
        return iter([cell.state() for cell in self.cell_list])

    def get_state_list(self):
        return [s for s in self.states()]

    def zero_states(self):
        for state in self.states():
            if isinstance(state, tuple):
                for s in state:
                    s.fill_(0.)
            else:
                state.fill_(0.)


class RNNCell(_RNNCellBase):

    def __init__(self, input_size, hidden_size, bias=True, nonlinearity="tanh", **kwargs):
        super(RNNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.nonlinearity = nonlinearity
        self.weight_ih = Parameter(torch.Tensor(hidden_size, input_size))
        self.weight_hh = Parameter(torch.Tensor(hidden_size, hidden_size))
        if bias:
            self.bias_ih = Parameter(torch.Tensor(hidden_size))
            self.bias_hh = Parameter(torch.Tensor(hidden_size))
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)
        self.reset_parameters()

    def check_forward(self, x, h):
        self.check_forward_input(x)
        self.check_forward_hidden(x, h)

    def forward(self, input, state):
        x = input 
        h = state
        self.check_forward(x, h)
        if self.nonlinearity == "tanh":
            func = self._backend.RNNTanhCell
        elif self.nonlinearity == "relu":
            func = self._backend.RNNReLUCell
        else:
            raise RuntimeError(
                "Unknown nonlinearity: {}".format(self.nonlinearity))
        state = output = func(
            x, h,
            self.weight_ih, self.weight_hh,
            self.bias_ih, self.bias_hh,
        )
        return output, state

    def register_state_buffer(self, batch_size):
        self.register_buffer('h', torch.Tensor(batch_size, self.hidden_size))

    def state(self):
        return self.h


class LSTMCell(_RNNCellBase):

    def __init__(self, input_size, hidden_size, bias=True, **kwargs):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.weight_ih = Parameter(torch.Tensor(4 * hidden_size, input_size))
        self.weight_hh = Parameter(torch.Tensor(4 * hidden_size, hidden_size))
        if bias:
            self.bias_ih = Parameter(torch.Tensor(4 * hidden_size))
            self.bias_hh = Parameter(torch.Tensor(4 * hidden_size))
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)
        self.reset_parameters()

    def check_forward(self, x, h, c):
        self.check_forward_input(x)
        self.check_forward_hidden(x, h, '[0]')
        self.check_forward_hidden(x, c, '[1]')

    def forward(self, input, state):
        x = input
        h, c = state
        check_forward(self, x, h, c)
        h_, c_ = self._backend.LSTMCell(
            input, hx,
            self.weight_ih, self.weight_hh,
            self.bias_ih, self.bias_hh,
        )
        output = h_
        state = (h_, c_)
        return output, state

    def register_state_buffer(self, batch_size):
        self.register_buffer('h', torch.Tensor(batch_size, self.hidden_size))
        self.register_buffer('c', torch.Tensor(batch_size, self.hidden_size))

    def state(self):
        return (self.h, self.c)


class FastWeightRNNCell(_RNNCellBase):
    """
    Fast weight RNN cell, with layer norm
    Ba et al. 2016
    """
    def __init__(self, input_size, hidden_size, bias=True, nonlinearity="relu", lam=1., eta=1., **kwargs):
        super(FastWeightRNNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.nonlinearity = nonlinearity
        self.lam = lam
        self.eta = eta

        self.rec_slow = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.inp_slow = nn.Linear(input_size, hidden_size, bias=bias)
        self.fast = FastWeight(hidden_size, lam=lam, eta=eta)
        self.ln = nn.LayerNorm(hidden_size)
        self.act = self.get_nonlinearity()

        self.reset_parameters()

    def check_forward(self, x, h, A):
        self.check_forward_input(x)
        self.check_forward_hidden(x, h)
        self.check_forward_fast_weight(x, h, A)

    def forward(self, input, state):
        x = input
        h, A = state
        self.check_forward(x, h, A)
        h_ = self.act(self.ln(self.rec_slow(h) + self.inp_slow(x)))
        A_ = self.fast(A, h_)
        output = h_
        state = (h_, A_)
        return output, state

    def register_state_buffer(self, batch_size):
        self.register_buffer('h', torch.Tensor(batch_size, self.hidden_size))
        self.register_buffer('A', torch.Tensor(batch_size, self.hidden_size, self.hidden_size))

    def state(self):
        return (self.h, self.A)


class FastWeightLSTMCell(_RNNCellBase):
    """
    Fast weight LSTM cell
    Keller et al. 2018
    """
    pass
    # [TODO] implement this



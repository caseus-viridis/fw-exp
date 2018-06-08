import torch
import torch.nn as nn
import numpy as np
from rnn_cells import RNNCell, LSTMCell, FastWeightRNNCell, FastWeightLSTMCell, RNNCellStack


def position_encoding(sentence_size, embedding_size):
    """
    Position Encoding described in section 4.1 of End to end Memory Networks
    """
    encoding = np.ones((embedding_size, sentence_size), dtype=np.float32)
    ls = sentence_size+1
    le = embedding_size+1
    for i in range(1, le):
        for j in range(1, ls):
            encoding[i-1, j-1] = (i - (embedding_size+1)/2) * (j - (sentence_size+1)/2)
    encoding = 1 + 4 * encoding / embedding_size / sentence_size
    # Make position encoding of time words identity to avoid modifying them 
    encoding[:, -1] = 1.0

    return np.transpose(encoding)

def create_rnn(cell_type, embed_size, hidden_size, num_layers, **rnn_config):
    if cell_type=='rnn':
        rnn_cell = RNNCell
    elif cell_type=='lstm':
        rnn_cell = LSTMCell
    elif cell_type=='fw-rnn':
        rnn_cell = FastWeightRNNCell
    elif cell_type=='fw-lstm':
        rnn_cell = FastWeightLSTMCell
    else:
        raise RuntimeError("unsupported RNN cell type {}".format(cell_type))
    return RNNCellStack([rnn_cell(embed_size if _i==0 else hidden_size, hidden_size, **rnn_config) for _i in range(num_layers)])


class ARTLearner(nn.Module):
    """
    Model that learns the associative retrieval task
    """
    def __init__(self, batch_size, seq_len, vocab_size, embed_size, hidden_size, num_layers=1, cell_type='fw-rnn', **rnn_config):
        super(ARTLearner, self).__init__()

        self.batch_size = batch_size
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.cell_type = cell_type

        self.build_model(**rnn_config)
        self.init_rnn_states()
    
    def build_model(self, **rnn_config):
        # embedding layer
        self.emb = nn.Embedding(self.vocab_size, self.embed_size)
        # multilayer RNN
        self.rnn = create_rnn(self.cell_type, self.embed_size, self.hidden_size, self.num_layers, **rnn_config)
        # final classifier, returns logits
        self.emt = nn.Sequential(
            nn.Linear(self.hidden_size, self.embed_size), 
            nn.ReLU(), 
            nn.Linear(self.embed_size, self.vocab_size)
        )

    def init_rnn_states(self):
        # register persistent buffers for all RNN states, and return handles
        self.rnn.register_state_buffers(self.batch_size)
        # zero their entries
        self.rnn.zero_states()

    def forward(self, input):
        states = self.rnn.get_state_list()
        for t in range(self.seq_len):
            x, states = self.rnn(self.emb(input[:, t]), states)
        return self.emt(x)

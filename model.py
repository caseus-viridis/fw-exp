import torch
import torch.nn as nn
import numpy as np
from fw_modules import FastWeightRNNCell, FastWeightLSTMCell


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


def ARTLearner(nn.Module):

    def __init__(self, seq_len, vocab_size, embed_size, hidden_size):
        super(ARTLearner, self).__init__()

        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size

        self.emb = nn.Embedding(vocab_size, embed_size)
        self.rnn = FastWeightRNNCell(embed_size, hidden_size)

    def forward(self, x):
        return x
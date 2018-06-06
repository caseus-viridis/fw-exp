import pickle
import os
import random
import numpy as np
import argparse
import torch
import torch.nn as nn
from data import ART
from fast_weight_modules import FastWeightRNNCell, FastWeightRNN

parser = argparse.ArgumentParser(description='Learn associative retrieval tasks')
parser.add_argument('--data-dir', default='./data', help='data directory (default: ./data)')
parser.add_argument('-t', '--task', type=int, default=1, help='type of task (in (1, 2, 3), default: 1)')
parser.add_argument('-n', '--num-pairs', type=int, default=8, help='number of K-V pairs (default: 8)')
parser.add_argument('-b', '--batch-size', type=int, default=128, help='batch size (default: 128)')
parser.add_argument('-e', '--epochs', type=int, default=1000, help='number of epochs (default: 1000)')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate (default: 0.0001)')
parser.add_argument('--lambda', type=float, default=1., help='lambda (default: 1.)')
parser.add_argument('--eta', type=float, default=1., help='eta (default: 1.)')
parser.add_argument('--seed', type=int, default=666, help='random seed (default: 666)')
parser.add_argument('--cpu', action='store_true', default=False, help='disables CUDA training')
args = parser.parse_args()
args.cuda = not args.cpu and torch.cuda.is_available()
if args.cuda:
    torch.cuda.manual_seed_all(args.seed)
else:
    torch.manual_seed(args.seed)

# ART data
dataset = ART(
    data_dir=args.data_dir,
    task=args.task,
    num_pairs=args.num_pairs,
    batch_size=args.batch_size,
    shuffle=True,
    cuda=args.cuda)

batch_size, input_size, hidden_size = 4, 10, 16
net = FastWeightRNNCell(input_size, hidden_size)
x = torch.randn(batch_size, input_size)
h = torch.rand(batch_size, hidden_size)
A = torch.zeros(batch_size, hidden_size, hidden_size)
h_, A_ = net(x, h, A)

def train(epoch):
    model.train()


def test():
    model.eval()
    for i, data in enumerate(dataset.test):
        x, y = data
        if args.cuda:
            x.cuda()
            y.cuda()
        
        

# data_file = 'associative-retrieval_task{}_{}pairs.pkl'.format(args.task, args.num_pairs)
# with open(os.path.join(args.data_dir, data_file), 'rb') as f:
#     data = pickle.load(f)

# word_idx = data['word_idx']
# vocab_size = len(word_idx)
# x_len = data['x_train'].shape[1]
# n_train = data['x_train'].shape[0]
# n_test = data['x_test'].shape[0]
# n_val = data['x_val'].shape[0]
# train_labels = np.argmax(data['y_train'], axis=1)
# test_labels = np.argmax(data['y_test'], axis=1)
# val_labels = np.argmax(data['y_val'], axis=1)

# print("x_train", data['x_train'][0])
# print("y_train", data['y_train'][0])
# print("Training set shape", data['x_train'].shape)
# print("Training Size", n_train)
# print("Validation Size", n_val)
# print("Testing Size", n_test)

# batches_train = zip(range(0, n_train - args.batch_size, args.batch_size), range(args.batch_size, n_train, args.batch_size))
# batches_train = [(start, end) for start, end in batches_train]
# batches_val = zip(range(0, n_val - args.batch_size, args.batch_size), range(args.batch_size, n_val, args.batch_size))
# batches_val = [(start, end) for start, end in batches_val]
# batches_test = zip(range(0, n_test - args.batch_size, args.batch_size), range(args.batch_size, n_test, args.batch_size))
# batches_test = [(start, end) for start, end in batches_test]

# batches = batches_train
# for t in range(1, args.epochs + 1):
#     random.Random(args.seed+t).shuffle(batches)
#     for b_idx_train, (start, end) in enumerate(batches):
#         x_train = data['x_train'][start:end]
#         x_train_lens = [x_len] * args.batch_size
#         y_train = data['y_train'][start:end]
#         import ipdb; ipdb.set_trace()

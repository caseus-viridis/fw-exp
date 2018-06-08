import pickle
import os
import random
import numpy as np
import argparse
import torch
import torch.nn as nn
from torch.optim import Adam
from data import ART
from model import ARTLearner

parser = argparse.ArgumentParser(description='Learn associative retrieval tasks')
parser.add_argument('--data-dir', default='./data', help='data directory (default: ./data)')
parser.add_argument('-t', '--task', type=int, default=1, help='type of task (in (1, 2, 3), default: 1)')
parser.add_argument('-k', '--num-pairs', type=int, default=4, help='number of K-V pairs (default: 4)')
parser.add_argument('-bs', '--batch-size', type=int, default=128, help='batch size (default: 128)')
parser.add_argument('-c', '--cell', type=str, default='fw-rnn', help='RNN cell type ("fw-rnn" (default), "rnn", "lstm", "fw-lstm")')
parser.add_argument('-es', '--embed-size', type=int, default=100, help='embedding size (default: 100)')
parser.add_argument('-hs', '--hidden-size', type=int, default=100, help='hidden size (default: 100)')
parser.add_argument('-l', '--layers', type=int, default=1, help='number of layers (default: 1)')
parser.add_argument('-e', '--epochs', type=int, default=100, help='number of epochs (default: 100)')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 1e-4)')
parser.add_argument('--lam', type=float, default=1., help='lambda (default: 1.)')
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
    shuffle_train=True
)
print(dataset)

# model
model = ARTLearner(
    batch_size=args.batch_size,
    seq_len=dataset.train.get_seq_len(),
    vocab_size=dataset.train.get_vocab_size(), 
    embed_size=args.embed_size,
    hidden_size=args.hidden_size,
    num_layers=args.layers,
    cell_type=args.cell,
    bias=True,
    nonlinearity='relu',
    lam=args.lam,
    eta=args.eta
)
if args.cuda:
    model.cuda()
print(model)

loss_func = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=args.lr)

# training and testing
def train(epoch):
    model.train()
    total_loss = 0.
    for batch, (x, y) in enumerate(dataset.train_loader):
        if args.cuda:
            x, y = x.cuda(), y.cuda()
        loss = loss_func(model(x), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print("Epoch {}: training loss = {:.4f}".format(epoch, total_loss/(batch+1)))
    val()

def val():
    model.eval()
    val_loss = correct = 0.
    with torch.no_grad():
        for batch, (x, y) in enumerate(dataset.val_loader):
            if args.cuda:
                x, y = x.cuda(), y.cuda()
            logits = model(x)
            val_loss += loss_func(logits, y).item()
            pred = logits.max(1, keepdim=True)[1]
            correct += pred.eq(y.view_as(pred)).sum().item()
    val_loss /= batch + 1
    correct /= (batch + 1) * args.batch_size
    print("\tvalidation loss = {:.4f}, correct = {:.2f}%".format(val_loss, correct*100))

def test():
    model.eval()
    test_loss = correct = 0.
    with torch.no_grad():
        for batch, (x, y) in enumerate(dataset.test_loader):
            if args.cuda:
                x, y = x.cuda(), y.cuda()
            logits = model(x)
            test_loss += loss_func(logits, y).item()
            pred = logits.max(1, keepdim=True)[1]
            correct += pred.eq(y.view_as(pred)).sum().item()
    test_loss /= batch + 1
    correct /= (batch + 1) * args.batch_size
    print("\ttest loss = {:.4f}, correct = {:.2f}%".format(test_loss, correct*100))
    
for epoch in range(args.epochs):
    train(epoch)

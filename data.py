import os
import pickle
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


NUM_EXAMPLES = {'train': 100000, 'val': 10000, 'test': 20000}  # sizes of datasets
KEYS = map(chr, range(97, 97 + 26))  # 26 letters
VALS = map(str, range(10))  # 10 digits
PROMPT = '?'
SEED = .666  # random seed for permutation in task3


def ensure_dir_exist(path):
    outdir = os.path.dirname(path)
    if outdir != '' and not os.path.isdir(outdir):
        os.makedirs(outdir)
    return path


def code_book(keys, vals, prompt):
    return dict(zip(keys + vals + [prompt], range(len(keys) + len(vals) + 1)))


def generate_art_example(num_pairs, keys, vals, prompt, task, shuffle_fn):
    nkeys, nvals = len(keys), len(vals)
    current_keys, current_vals, current_pairs = [], [], {}
    # sample from task example space
    for i in range(0, num_pairs):
        key = keys[random.randint(0, nkeys - 1)]
        while key in current_pairs.keys():
            key = keys[random.randint(0, nkeys - 1)]
        val = vals[random.randint(0, nvals - 1)]
        current_pairs[key] = val
        current_keys += key
        current_vals += val
    query = random.choice(list(current_pairs.keys()))
    target = current_pairs[query]
    # synthesize the sequence
    if task == 1:
        x = [item for pair in zip(current_keys, current_vals) for item in pair]
    elif task == 2:
        x = current_keys + current_vals
    elif task == 3:
        shuffle_fn(current_vals)
        x = current_keys + current_vals
    x += [prompt, prompt, query]
    return x, target


def encode_art_example(x, y, word_idx):
    x_enc = np.array([word_idx[w] for w in x])
    # y_onehot = np.zeros([len(word_idx)])
    # y_onehot[word_idx[y]] = 1
    y_enc = word_idx[y]
    return x_enc, y_enc


def generate_art_data(num_examples, num_pairs, keys, vals, prompt, task, shuffle_fn):
    word_idx = code_book(keys, vals, prompt)
    x = np.zeros([num_examples, num_pairs * 2 + 3], dtype=np.int64)
    y = np.zeros(num_examples, dtype=np.int64)
    for i in range(num_examples):
        _x, _y = generate_art_example(num_pairs, keys, vals, prompt, task, shuffle_fn)
        x[i], y[i] = encode_art_example(_x, _y, word_idx)
    return {'x': x, 'y': y, 'word_idx': word_idx, 'k': num_pairs, 'n': num_examples, 't': task}


def load_art_data(data_dir, task, num_pairs, dset):
    data_file = "{}/associative-retrieval-task{}_{}pairs_{}.pkl".format(
        ensure_dir_exist(data_dir), task, num_pairs, dset)
    if os.path.exists(data_file):
        print("Loading ART data from {}".format(data_file))
        with open(data_file, 'rb') as f:
            data = pickle.load(f)
    else:
        print("Generating ART data: {} examples of task {} of {} pairs".format(NUM_EXAMPLES[dset], task, num_pairs))
        data = generate_art_data(
            NUM_EXAMPLES[dset],
            num_pairs,
            keys=KEYS,
            vals=VALS,
            prompt=PROMPT,
            task=task,
            shuffle_fn=lambda ls: random.shuffle(ls, lambda: SEED))
        print("Dumping ART data to {}".format(data_file))
        with open(data_file, 'wb') as f:
            pickle.dump(data, f, protocol=2)
    return data


class ARTDataset(Dataset):

    def __init__(self, data_dir='./data', task=1, num_pairs=8, dset='train'):
        self.data = load_art_data(data_dir, task, num_pairs, dset)

    def __len__(self):
        return self.data['n']

    def __getitem__(self, idx):
        assert idx < self.__len__(), "index {} out of bound ({})".format(idx, self.__len__())
        return (self.data['x'][idx], self.data['y'][idx])

    def get_word_idx(self):
        return self.data['word_idx']

    def get_vocab_size(self):
        return len(self.get_word_idx())
    
    def get_seq_len(self):
        return self.data['k'] * 2 + 3


class ART(object):

    def __init__(self, data_dir='./data', task=1, num_pairs=8, batch_size=64, shuffle=True):
        self.train = ARTDataset(data_dir, task, num_pairs, dset='train')
        self.val = ARTDataset(data_dir, task, num_pairs, dset='val')
        self.test = ARTDataset(data_dir, task, num_pairs, dset='test')
        self.train_loader = DataLoader(self.train, shuffle=shuffle, batch_size=batch_size, drop_last=True)
        self.val_loader = DataLoader(self.val, batch_size=batch_size, drop_last=True)
        self.test_loader = DataLoader(self.test, batch_size=batch_size, drop_last=True)

import numpy as np
import random
import pickle
import os
import argparse

parser = argparse.ArgumentParser(description='Generate data for associative retrieval task')
parser.add_argument('--data-dir', default='./data', help='data directory (default: ./data)')
parser.add_argument('-t', '--task', type=int, default=1, help='type of task (in (1, 2, 3), default: 1)')
parser.add_argument('-n', '--num-pairs', type=int, default=25, help='number of K-V pairs (default: 25)')
parser.add_argument('--num-train', type=int, default=100000, help='number of training examples (default: 100000)')
parser.add_argument('--num-val', type=int, default=10000, help='number of validation examples (default: 10000)')
parser.add_argument('--num-test', type=int, default=20000, help='number of test examples (default: 20000)')
parser.add_argument('--seed', type=int, default=0, help='random seed (default: 0)')
parser.add_argument(
    '--shuffle-seed',
    type=float,
    default=0.666,
    help='random seed for shuffling (float between 0. and 1., default: 0.666)')
args = parser.parse_args()

shuffle_fn = lambda ls: random.shuffle(ls, lambda: args.shuffle_seed)
random.seed(args.seed)
x_len = args.num_pairs * 2 + 3
vocab_size = 26 + 10 + 1  # 26 characters (keys) + 10 digits (values) + '?' prompt

# Add characters to vocab
word_idx = {chr(ord('a') + i): i for i in range(26)}
# Add numbers to vocab
for i in range(10):
    word_idx[str(i)] = 26 + i
# Add prompt to vocab
word_idx['?'] = 36


def generate_example(word_idx, num_pairs, task_type, shuffle_fn):
    keys, vals = [], []
    current_pairs = {}

    # Sample from task example space
    for i in range(0, num_pairs):
        key = chr(ord('a') + random.randint(0, 25))
        while key in current_pairs.keys():
            key = chr(ord('a') + random.randint(0, 25))
        val = str(random.randint(0, 9))
        current_pairs[key] = val
        keys.append(key)
        vals.append(val)
    query = random.choice(list(current_pairs.keys()))
    target = current_pairs[query]

    # Synthesize the sequence
    if task_type == 1:
        x = [item for pair in zip(keys, vals) for item in pair]
    elif task_type == 2:
        x = keys + vals
    elif task_type == 3:
        shuffle_fn(vals)
        x = keys + vals
    x += ['?', '?', query]

    # Encode x using word_idx, and y as onehot
    x_enc = np.array([word_idx[w] for w in x])
    y_onehot = np.zeros([vocab_size])
    y_onehot[word_idx[target]] = 1

    return x_enc, y_onehot


# Generate and dump data
x_train = np.zeros([args.num_train, x_len], dtype=np.float32)
x_val = np.zeros([args.num_val, x_len], dtype=np.float32)
x_test = np.zeros([args.num_test, x_len], dtype=np.float32)
y_train = np.zeros([args.num_train, vocab_size], dtype=np.float32)
y_val = np.zeros([args.num_val, vocab_size], dtype=np.float32)
y_test = np.zeros([args.num_test, vocab_size], dtype=np.float32)

for i in range(0, args.num_train):
    x_train[i], y_train[i] = generate_example(word_idx, args.num_pairs, args.task, shuffle_fn)
for i in range(0, args.num_test):
    x_test[i], y_test[i] = generate_example(word_idx, args.num_pairs, args.task, shuffle_fn)
for i in range(0, args.num_val):
    x_val[i], y_val[i] = generate_example(word_idx, args.num_pairs, args.task, shuffle_fn)

d = {
    'x_train': x_train,
    'x_test': x_test,
    'x_val': x_val,
    'y_train': y_train,
    'y_test': y_test,
    'y_val': y_val,
    'word_idx': word_idx
}

data_file = 'associative-retrieval_task{}_{}pairs.pkl'.format(args.task, args.num_pairs)
with open(os.path.join(args.data_dir, data_file), 'wb') as f:
    pickle.dump(d, f, protocol=2)
print("Generated data dumped to {}".format(data_file))

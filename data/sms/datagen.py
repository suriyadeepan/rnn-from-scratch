SMS_FILENAME = 'data/sms/sms.txt'
MADURAI_FILENAME = 'data/madurai/sample.txt'
KERNEL_FILENAME = 'data/linux/linux_kernel_3Maa'

MADURAI_PATH = 'data/madurai/'
SMS_PATH = 'data/sms/'
KERNEL_PATH = 'data/linux/'


import csv
import numpy as np
import pickle as pkl



def read_lines_sms(filename):
    with open(filename, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        return [ row[-1] for row in list(reader) ]

def read_lines(filename):
    with open(filename) as f:
        return f.read().split('\n')

def index_(lines):
    vocab = list(set('\n'.join(lines)))
    ch2idx = { k:v for v,k in enumerate(vocab) }
    return vocab, ch2idx

def to_array(lines, seqlen, ch2idx):
    # combine into one string
    raw_data = '\n'.join(lines)
    num_chars = len(raw_data)
    # calc data_len
    data_len = num_chars//seqlen
    # create numpy arrays
    X = np.zeros([data_len, seqlen])
    Y = np.zeros([data_len, seqlen])
    # fill in
    for i in range(0, data_len):
        X[i] = np.array([ ch2idx[ch] for ch in raw_data[i*seqlen:(i+1)*seqlen] ])
        Y[i] = np.array([ ch2idx[ch] for ch in raw_data[(i*seqlen) + 1 : ((i+1)*seqlen) + 1] ])
    # return ndarrays
    return X, Y

def process_data(path, filename, seqlen=10):
    lines = read_lines(filename)
    idx2ch, ch2idx = index_(lines)
    X, Y = to_array(lines, seqlen, ch2idx)
    np.save(path+ 'idx_x.npy', X)
    np.save(path+ 'idx_y.npy', Y)
    with open(path+ 'metadata.pkl', 'wb') as f:
        pkl.dump( {'idx2ch' : idx2ch, 'ch2idx' : ch2idx }, f )


if __name__ == '__main__':
    process_data(path = KERNEL_PATH,
            filename = KERNEL_FILENAME)



def load_data(path):
    # read data control dictionaries
    with open(path + 'metadata.pkl', 'rb') as f:
        metadata = pkl.load(f)
    # read numpy arrays
    X = np.load(path + 'idx_x.npy')
    Y = np.load(path + 'idx_y.npy')
    return X, Y, metadata['idx2ch'], metadata['ch2idx'], X.shape[1]

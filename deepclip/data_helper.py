import numpy as np
from sklearn.utils import class_weight
import gzip

__all__ = ['dataLoader']

SEQ_MAP = {
        "A": [1, 0, 0, 0], 
        "T": [0, 1, 0, 0], 
        "C": [0, 0, 1, 0], 
        "G": [0, 0, 0, 1], 
        "N": [0.25, 0.25, 0.25, 0.25]
        }
def nucleotides_to_onehot(seq):
    return [SEQ_MAP[c] for c in seq]

class dataLoader(object):
    def __init__(self, file, pad = 27, prob = 0.0):
        self.file = file
        self.pad = pad
        self.prob = prob
        self._load_file()

    def _load_file(self):
        with gzip.open(self.file, 'rt') as f:
            x, y = [], []
            for line in f:
                line = line.rstrip('\n').split('\t')
                seq, _, _, label = line[0], line[1], line[2], line[3]
                if self.pad != 0:
                    seq = seq + 'N' * self.pad
                seq_onehot = nucleotides_to_onehot(seq)
                x.append(seq_onehot)
                y.append(int(label))
        self.x = np.array(x)
        self.y = np.array(y)

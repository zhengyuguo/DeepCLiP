import numpy as np
from sklearn.utils import class_weight
import gzip

__all__ = ['dataLoader']

SEQ_MAP = {
        "A": [1, 0, 0, 0], 
        "T": [0, 1, 0, 0], 
        "C": [0, 0, 1, 0], 
        "G": [0, 0, 0, 1], 
        "N": [0, 0, 0, 0]
        }
def nucleotides_to_onehot(seq):
    return [SEQ_MAP[c] for c in seq]

def add_noise(seq, prob = 0.05):
    return ''.join([np.random.choice(['A','T','C','G']) if np.random.sample() < prob else c for c in seq]) 

class dataLoader(object):
    def __init__(self, file, pad = 27, prob = 0.75):
        self.file = file
        self.pad = pad
        self.prob = prob
        self._load_file()

    def _load_file(self):
        with gzip.open(self.file, 'rt') as f:
            x, y = [], []
            x_n = []
            for line in f:
                line = line.rstrip('\n').split('\t')
                seq, _, _, label = line[0], line[1], line[2], line[3]
                noisy_seq = add_noise(seq, self.prob)
                if self.pad != 0:
                    seq = seq + 'N' * self.pad
                    noisy_seq = noisy_seq + 'N' * self.pad
                seq_onehot = nucleotides_to_onehot(seq)
                noisy_seq_onehot = nucleotides_to_onehot(noisy_seq)
                x.append(seq_onehot)
                x_n.append(noisy_seq_onehot)
                y.append(int(label))
        self.x = np.array(x)
        self.y = np.array(y)
        self.x_n = np.array(x_n)

#DOT_MAP = {
#        ".": [1, 0, 0], 
#        "(": [0, 1, 0], 
#        ")": [0, 0, 1],
#        "N": [0, 0, 0]
#        }
#
#STRUCT_MAP = {
#        "s": [1, 0, 0, 0, 0, 0], 
#        "f": [0, 1, 0, 0, 0, 0], 
#        "h": [0, 0, 1, 0, 0, 0], 
#        "i": [0, 0, 0, 1, 0, 0], 
#        "t": [0, 0, 0, 0, 1, 0],
#        "m": [0, 0, 0, 0, 0, 1],
#        "N": [0, 0, 0, 0, 0, 0]
#        }
#
#MAP = {'A':1,'T':2,'C':3,'G':4,'N':0}
#def nucleotides_to_idx(seq):
#    return [MAP[c] for c in seq]
#
#MAP_D = {'.':1,'(':2,')':3, 'N':0}
#def dotbrac_to_idx(seq):
#    return [MAP_D[c] for c in seq]
#
#def dotbrac_to_onehot(seq):
#    return [DOT_MAP[c] for c in seq]
#
#def structure_to_onehot(seq):
#    return [STRUCT_MAP[c] for c in seq]
#
#class dataLoader(object):
#    def __init__(self, file, pad = 27):
#        self.file = file
#        self.pad = pad
#        self._load_file()
#
#    def _load_file(self):
#        with gzip.open(self.file, 'rt') as f:
#            x1, x2, x3, x4, x5, y = [], [], [], [], [], []
#            for line in f:
#                line = line.rstrip('\n').split('\t')
#                seq, dotbrac, struct, label = line[0], line[1], line[2], line[3]
#                if self.pad != 0:
#                    seq = seq + 'N'*self.pad
#                    dotbrac = dotbrac + 'N'*self.pad
#                    struct = struct + 'N'*self.pad
#
#                seq_idx = nucleotides_to_idx(seq)
#                seq_onehot = nucleotides_to_onehot(seq)
#                dot_idx = dotbrac_to_idx(dotbrac)
#                dot_onehot = dotbrac_to_onehot(dotbrac)
#                struct_onehot = structure_to_onehot(struct)
#                x1.append(seq_idx)
#                x2.append(seq_onehot)
#                x3.append(dot_idx)
#                x4.append(dot_onehot)
#                x5.append(struct_onehot)
#                y.append(int(label))
#
#        self.x1 = np.array(x1)
#        self.x2 = np.array(x2)
#        self.x3 = np.array(x3)
#        self.x4 = np.array(x4)
#        self.x5 = np.array(x5)
#        self.y = np.array(y)
#

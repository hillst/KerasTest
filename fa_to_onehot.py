#!/usr/bin/env python
import sys
import numpy as np
base_to_onehot = {"A": [1,0,0,0], "T": [0,1,0,0], "C": [0,0,1,0], "G": [0,0,0,1]}
idx_to_base = {0: "A", 1 : "T", 2 : "C", 3: "G"}

def main():
    seq = load_fasta(sys.argv[1])

def load_fasta(fname):
    seqs = []
    seq = []
    for line in open(fname, 'r'):
        if ">" in line:
            if seq != []:
                seqs.append(seq)
            else:
                seq = []
        else:
            for char in line.strip():
                seq.append(base_to_onehot(char))
        
    seqs.append(seq)
    return seqs

def base_to_onehot(base):
    if base == "A":
        return np.asarray([1,0,0,0], dtype=np.float32)
    if base == "T":
        return np.asarray([0,1,0,0], dtype=np.float32)
    if base == "C":
        return np.asarray([0,0,1,0], dtype=np.float32)
    if base == "G":
        return np.asarray([0,0,0,1], dtype=np.float32)

if __name__ == "__main__":
    main()

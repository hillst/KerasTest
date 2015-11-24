#!/usr/bin/env python
import sys
from fa_to_onehot import load_fasta, idx_to_base
from keras.datasets.data_utils import get_file

from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.core import Dense, Activation, Dropout, TimeDistributedDense
from keras.layers.recurrent import LSTM

import sys
import numpy as np
ALPHABET_SIZE = 4
# cut the text in semi-redundant sequences of maxlen characters
seqs = load_fasta("test.fa")
print len(seqs)

maxlen = max([len(seq) for seq in seqs])

step = 1
X_train = [] #sentences to generate
y_train  = []

for seq in seqs:
	X_train.append(np.asarray(seq[1:]))
	y_train.append(np.asarray(seq[:-1]))





# build the model: 2 stacked LSTM
print('Build model...')
max_features = (4) #samples x length
embedding_size=(4) #samples x length x dim
model = Sequential()
#model.add(Embedding(max_features, embedding_size, mask_zero=True))
#model.add(Embedding(input_shape=(maxlen, 4), output_dim=(maxlen, 4), mask_zero=True, init="uniform"))
model.add(LSTM(int(sys.argv[1]), return_sequences=True, input_shape=(maxlen, 4)))
model.add(Dropout(0.2))
model.add(TimeDistributedDense(4))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

import time
starttime = time.time()
model.fit(np.asarray(X_train), np.asarray(y_train), batch_size=1, nb_epoch=20 )
endtime = time.time()
times.append(endtime - starttime)
from scipy import mean
print mean(times), model.evaluate(X_train, y_train)

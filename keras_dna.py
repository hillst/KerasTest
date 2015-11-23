#!/usr/bin/env python
import sys
from fa_to_onehot import load_fasta, idx_to_base
from keras.datasets.data_utils import get_file

from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
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
    for i in range(1, len(seq)):
        padding = np.zeros(((maxlen - i) , 4))
        sub_list = np.concatenate([padding, np.asarray(seq[:i])])
        #sub_list = np.asarray(seq[:i])
        X_train.append(sub_list)
        y_train.append(np.asarray(seq[i]))

print "Shape of DNA :", np.asarray(X_train).shape       





# build the model: 2 stacked LSTM
print('Build model...')
max_features = (4) #samples x length
embedding_size=(4) #samples x length x dim
model = Sequential()
#model.add(Embedding(max_features, embedding_size, mask_zero=True))
#model.add(Embedding(input_shape=(maxlen, 4), output_dim=(maxlen, 4), mask_zero=True, init="uniform"))
model.add(LSTM(32, return_sequences=False, input_shape=(maxlen, 4)))
model.add(Dropout(0.2))
model.add(Dense(4))
model.add(Activation('softmax'))

print "Compile model"
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

print "fitting"
model.fit(np.asarray(X_train), np.asarray(y_train), batch_size=1, nb_epoch=1)
print model.predict(np.asarray(X_train))

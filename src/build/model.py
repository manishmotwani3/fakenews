# First experiment for classifying news articles into 4 categories : Satire, Hoax, Truthful, Propaganda

import sys
import os
from nltk.tokenize import RegexpTokenizer
import numpy as np

import torch
from torch.autograd import Variable
import torch.autograd as autograd
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from LSTM import LSTM
from data_utils import create_dict, prep_minibatch

X = np.load('../../dataset/ds1/X_train.npy')
y = np.load('../../dataset/ds1/y_train.npy')

X_valid = np.load('../../dataset/ds1/X_valid.npy')
y_valid = np.load('../../dataset/ds1/y_valid.npy')

# Creating requisite dictionaries. No need dict for tags

corpus, tags, vocab, dictt, idx2word = create_dict(X, y)


# Parameters of the model

Hidden_dim = 25
Embed_dim = 25
epochs = 5
mini_batch_size = 20
no_of_batches = int(len(corpus)/mini_batch_size)

model = LSTM(Embed_dim, Hidden_dim, vocab, mini_batch_size)

loss_fn = torch.nn.SoftMarginLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 1e-2)



# Train the model

for e in range(epochs):

	print ("Epoch No. : ", e)
	total_loss = 0


	for i in range(no_of_batches):

		print ("Batch No. : ", i)
		sentences, targets, seq_lens = prep_minibatch(mini_batch_size, mini_batch_size*i, corpus, tags, dictt)

		#sentence = corpus[i]
		#tag = tags[i]
		#X, Y = Prep_x_y(sentence, tag)

		#sentence, tag = autograd.Variable(torch.from_numpy(X)), autograd.Variable(torch.FloatTensor(Y))


		model.zero_grad()


		model.hidden = model.init_hidden(mini_batch_size)

		tag_scores = model(sentences, seq_lens)

		loss = loss_fn(tag_scores, targets)


		loss.backward()
		optimizer.step()
 		total_loss += loss.data


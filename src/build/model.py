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
import matplotlib.pyplot as plt

import pickle

X = np.load('../../dataset/ds1/X_train.npy')
y = np.load('../../dataset/ds1/y_train.npy')

X_valid = np.load('../../dataset/ds1/X_valid.npy')
y_valid = np.load('../../dataset/ds1/y_valid.npy')

X = X[:1000]
y = y[:1000]

X_valid = X_valid[:200]
y_valid = y_valid[:200]


# Creating requisite dictionaries. No need dict for tags

corpus, tags, vocab, dictt, idx2word = create_dict(X, y)

valid_corpus, valid_tags = X_valid, y_valid

# Parameters of the model

Hidden_dim = 100
Embed_dim = 300
epochs = 20
mini_batch_size = 50
no_of_batches = int(len(corpus)/mini_batch_size)

model = LSTM(Embed_dim, Hidden_dim, vocab, mini_batch_size)
print ("Pre training")
model.pretrained_embs(dictt, idx2word)

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 1e-2)


valid_acc = []
valid_los = []

train_los = []
train_acc = []


def Valid_err():
	valid_err = 0
	acc = 0

	for i in range(int(len(valid_corpus)/mini_batch_size)):

		sentences, targets, seq_lens = prep_minibatch(mini_batch_size, mini_batch_size*i, valid_corpus, valid_tags, dictt)


		model.hidden = model.init_hidden(mini_batch_size)
		tag_scores = model(sentences, seq_lens)

		np_tag_scores = tag_scores.data.numpy()

		for j in range(len(np_tag_scores)):
			if (int(np.argmax(np_tag_scores, axis=1)[j])  ==  int(targets.data.numpy()[j])):
				acc += 1.0

		loss = loss_fn(tag_scores, targets)
		valid_err += loss.data
	return (valid_err.numpy()[0], acc)


# Train the model

for e in range(epochs):

	print ("Epoch No. : ", e)
	total_loss = 0
	tr_acc = 0.0

	for i in range(no_of_batches):

		print ("Batch No. : ", i)
		sentences, targets, seq_lens = prep_minibatch(mini_batch_size, mini_batch_size*i, corpus, tags, dictt)

		model.zero_grad()


		model.hidden = model.init_hidden(mini_batch_size)

		tag_scores = model(sentences, seq_lens)

		np_tag_scores = tag_scores.data.numpy()

		for j in range(len(np_tag_scores)):

			#print (int(np.argmax(np_tag_scores, axis=1)[j]), int(targets.data.numpy()[j]))

			if (int(np.argmax(np_tag_scores, axis=1)[j])  ==  int(targets.data.numpy()[j])):
				tr_acc += 1.0

		loss = loss_fn(tag_scores, targets)


		loss.backward()
		optimizer.step()
 		total_loss += loss.data

 	pickle.dump( model, open( "model1.p", "wb" ) )

 	valid_loss, acc = Valid_err()

 	valid_acc.append(float(acc/(int(len(valid_corpus)/mini_batch_size)*mini_batch_size)))
 	valid_los.append(valid_loss/int(len(valid_corpus)))

 	train_los.append((total_loss.numpy()[0])/int(len(corpus)))
 	train_acc.append(float(tr_acc/(int(len(corpus)/mini_batch_size)*mini_batch_size)))

 	np.save('valid_acc1.npy', valid_acc)
	np.save('valid_los1.npy', valid_los)

	np.save('train_los1.npy', train_los)
	np.save('train_acc1.npy', train_acc)

plt.plot(train_los)
#plt.plot(valid_los)

#plt.plot(valid_acc)
plt.title("Loss per sentence vs epochs")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.show()


plt.plot(train_acc)

#plt.plot(valid_acc)

#plt.plot(valid_acc)
plt.title("Accuracy per sentence vs epochs")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.show()

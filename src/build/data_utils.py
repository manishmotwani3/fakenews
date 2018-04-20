# This file includes the data utilities that are required for the experiments.

import os
import sys

import torch
from torch.autograd import Variable
import torch.autograd as autograd
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# Create dictionary for the vocabulary

def create_dict(X, y):

	corpus = []
	tags = []

	dictt = {}
	idx2word = ['zero_embed']
	vocab = 1

	for i in range(len(X)):

		for word in X[i]:
			if (word not in dictt):
				dictt[word] = vocab
				idx2word.append(word)
				vocab += 1

		corpus.append(X[i])

	dictt['OOV'] = vocab
	idx2word.append(word)
	vocab += 1

	return (corpus, y, vocab, dictt, idx2word)

# Function to prepare a minibatch ready for training

def prep_minibatch(mini_batch_size, i, corpuss, tagss, dictt):

	maxlen = max([len(x) for x in corpuss[i:i+mini_batch_size]])
	input_tensor = []
	output_tensor = []
	seq_l = []
	for k in range(mini_batch_size):
		tempp = corpuss[i+k]
		temp = []
		for l in range(len(tempp)):
			if(tempp[l] not in dictt):
				temp.append(dictt["OOV"])
			else:
				temp.append(dictt[tempp[l]])



		seq_l.append(len(corpuss[i+k]))
		for l in range(maxlen-len(corpuss[i+k])):
			temp.append(0)


		input_tensor.append(temp)


		output_tensor.append(tagss[i+k])

	input_tensor = [x for _, x in sorted(zip(seq_l,input_tensor), key=lambda pair: pair[0])]
	output_tensor = [x for _, x in sorted(zip(seq_l,output_tensor), key=lambda pair: pair[0])]
	seq_l.sort()
	input_tensor = input_tensor[::-1]
	output_tensor = output_tensor[::-1]
	seq_l = seq_l[::-1]

	batch_in = Variable(torch.LongTensor(input_tensor))
	batch_out = Variable(torch.LongTensor(output_tensor))

	return (batch_in, batch_out, seq_l)
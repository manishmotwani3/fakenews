# This file holds the LSTM class required for news classification


import torch
from torch.autograd import Variable
import torch.autograd as autograd
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from nltk.tokenize import RegexpTokenizer
import numpy as np

from gensim.models import KeyedVectors


# Class for LSTM 

class LSTM(torch.nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, mini_batch_size):
        super(LSTM, self).__init__()

        self.hidden_dim = hidden_dim

        self.word_embeddings = torch.nn.Embedding(vocab_size, embedding_dim)

        self.lstm = torch.nn.LSTM(embedding_dim, hidden_dim)

        self.hidden2tag = torch.nn.Linear(hidden_dim, 4)
        self.hidden = self.init_hidden(mini_batch_size)

    def init_hidden(self, mini_batch_size):

        return (autograd.Variable(torch.zeros(1, mini_batch_size, self.hidden_dim)),
                autograd.Variable(torch.zeros(1, mini_batch_size, self.hidden_dim)))

    def pretrained_embs(self, dictt, idx2word):

        word_vectors = KeyedVectors.load_word2vec_format('../../dataset/GoogleNews-vectors-negative300.bin', binary=True) 

        self.word_embeddings.weight.requires_grad=False

        for i in range(len(idx2word)):
            print (i)
            if (idx2word[i] != 'zero_embed'):
                self.word_embeddings[i] = word_vectors.get_vector(idx2word[i])

            #self.word_embeddings[i] = word_vectors.get_vector(idx2word[i])

        



    def forward(self, sentences, seq_lens):

        embeds = self.word_embeddings(sentences)

        embeds = embeds.transpose(0,1)

        embeds = pack_padded_sequence(embeds, np.array(seq_lens))

        lstm_out, self.hidden = self.lstm(
            embeds, self.hidden)

        lstm_out, _ = pad_packed_sequence(lstm_out)

        out_vals= autograd.Variable(torch.zeros(lstm_out.shape[1],lstm_out.shape[2]))
        for i, val in enumerate(seq_lens):
            out_vals[i] = lstm_out[val-1][i]
        out_vals = self.hidden2tag(out_vals)


        tag_scores = out_vals

        return tag_scores
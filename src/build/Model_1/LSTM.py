# This file holds the LSTM class required for news classification


import torch
from torch.autograd import Variable
import torch.autograd as autograd
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from nltk.tokenize import RegexpTokenizer
import numpy as np


# Class for LSTM 

class LSTM(torch.nn.Module):

    def __init__(self, embedDim, outputDim, hiddenDim, vocabSize, batchSize):
        super(LSTM, self).__init__()

        self.embedDim   = embedDim
        self.hiddenDim  = hiddenDim
        self.vocabSize  = vocabSize
        self.batchSize  = batchSize
        self.outputDim  = outputDim

        self.wordEmbeddings = torch.nn.Embedding(self.vocabSize, self.embedDim)
        self.lstm           = torch.nn.LSTM(self.embedDim, self.hiddenDim)
        self.hidden2tag     = torch.nn.Linear(self.hiddenDim, self.outputDim)

        self.lossFunc    = torch.nn.CrossEntropyLoss()
        self.optimizer  = torch.optim.Adam([{'params': self.lstm.parameters()},
                                            {'params': self.hidden2tag.parameters()}
                                            ], lr=0.01)

    def __str__(self):
        printStr = ""
        printStr += "-----------------LSTM Model Parameters-----------------------------" + "\n"
        printStr += "embedDim:\t" + str(self.embedDim) + "\n"
        printStr += "hiddenDim:\t" + str(self.hiddenDim) + "\n"
        printStr += "vocabSize:\t" + str(self.vocabSize) + "\n"
        printStr += "batchSize:\t" + str(self.batchSize) + "\n"
        printStr += "outputDim:\t" + str(self.outputDim) + "\n"
        
        printStr += "optimizer:\t" + str(self.optimizer) + "\n"
        printStr += "lossFunc:\t" + str(self.lossFunc) + "\n"
        printStr += "Model Components" + "\n"
        printStr += str(self.lstm) + "\n"
        printStr += str(self.wordEmbeddings) + "\n"
        printStr += str(self.hidden2tag) + "\n"
        printStr += "-------------------------------------------------------------------" + "\n"
        return printStr

    def forward(self, sentences, seq_lens):

        # Get embeddings for all words in the sentence
        embeds = self.wordEmbeddings(sentences)
        embeds = embeds.transpose(0,1)

        # Pack the output to send it to lstm
        embeds = pack_padded_sequence(embeds, np.array(seq_lens))


        lstm_out, _ = self.lstm(embeds)

        # Unpack the output obtained from lstm
        lstm_out, _ = pad_packed_sequence(lstm_out)

        # Get output for each sentence from lstm output
        out_vals= autograd.Variable(torch.zeros(lstm_out.shape[1],lstm_out.shape[2]))
        for i, val in enumerate(seq_lens):
            out_vals[i] = lstm_out[val-1][i]

        # Compute n-way class (un-normalized scores) by passing output of LSTM hidden unit through a linear layer
        tag_scores = self.hidden2tag(out_vals)

        return tag_scores
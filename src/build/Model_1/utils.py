# This file includes the data utilities that are required for the experiments.

import pickle, math, logging
import torch
from nltk.tokenize import RegexpTokenizer
from torch.autograd import Variable

# Reads data stored in file one line at a time, tokenizes each line to a list of words
# Returns a list of sentences, where each sentence is a list of words
def read_X(filename):

		X = []
		with open(filename,"r") as f:
			for line in f:
				tokenizer = RegexpTokenizer(r'\w+')
				tokens = tokenizer.tokenize(line.lower())
				X.append(tokens)

		return X

# Reads labels stored in file, converts them to integers
# Returns a list of labels
def read_y(filename):

	y = []
	with open(filename,"r") as f:
		for line in f:
			y.append(int(line))

	return y

# Prepare batch out of corpus to be sent to LSTM unit
def prepBatch(batchSize, startIdx, corpus, tags, wordVocab):

	endIdx 		= min(startIdx + batchSize, len(corpus))
	maxSentLen 	= max([len(x) for x in corpus[startIdx : endIdx]])
	inputTensor 	= []
	outputTensor 	= []
	seqLens 		= [] # Stores lengths of all sequences in this batch
	for index in range(startIdx,endIdx):
		sequence = corpus[ index ] # Current sequence of words being processed
		sent2Idx = []
		# Convert words in sequence to index
		for word in sequence:
			if(word not in wordVocab):
				sent2Idx.append(wordVocab["OOV"])
			else:
				sent2Idx.append(wordVocab[word])

		sent2Idx = sent2Idx + [0]*(maxSentLen - len(sequence)) # Append zeros upto maxSentLen	
		seqLens.append(len(sequence))
		inputTensor.append(sent2Idx)
		outputTensor.append(tags[ index ])
		

	# Sort data in decreasing order of length
	sortedData 		= sorted(zip(seqLens,inputTensor, outputTensor), key=lambda pair: pair[0], reverse=True)
	seqLens 		= [x for x,_,_ in sortedData ]
	inputTensor 	= [x for _,x,_ in sortedData ]
	outputTensor 	= [int(x) for _,_,x in sortedData ]
	
	batchIn 	= Variable(torch.LongTensor(inputTensor))
	batchOut 	= Variable(torch.LongTensor(outputTensor))

	return (batchIn, batchOut, seqLens)

if __name__ == '__main__':
	pass
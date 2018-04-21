
# self.wordVocab,self.index2Word = self.createWordVocabulary(self.trainData) 
# LSTM model

# First experiment for classifying news articles into 4 categories : Satire, Hoax, Truthful, Propaganda

import pickle, argparse,itertools, math
import sys,os,time
import logging
from nltk.tokenize import RegexpTokenizer
import numpy as np

import torch
from torch.autograd import Variable
import torch.autograd as autograd
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from LSTM import LSTM
from gensim.models import KeyedVectors

from trainer import Trainer
from utils import prepBatch, read_X, read_y

class FourToTwo(object):
	"""docstring for FourToTwo"""
	def __init__(self, paramDict):
		super(FourToTwo, self).__init__()
		self.resultDir 	= paramDict["resultDir"]
		self.dataDir 	= paramDict["dataDir"]
		self.modelFile 	= paramDict["modelFile"]

		
		logging.basicConfig(filename=self.resultDir + "/log.txt",level=logging.DEBUG) # Get logger object
		# Add console to the logger, with level INFO so that INFO logs go to both console and file, while DEBUG logs go to file only
		self.logger = logging.getLogger()
		console = logging.StreamHandler()
		console.setLevel(logging.INFO)
		logging.getLogger('').addHandler(console)

		self.logger.info("Loading test data...")
		self.testData 	= read_X(self.dataDir + '/X_1000.txt')
		self.testLabel 	= read_y(self.dataDir + '/y_1000.txt')

		self.logger.info("Reading saved model...")
		self.model, self.wordVocab, self.index2Word	=  pickle.load(open(self.modelFile,"rb"))
		self.weights	= np.random.rand(4,1)
		# self.weights	= [0.5,0.5,0.5,0.5]
		self.numClass 	= 2

		self.logger.info(self)

	def __str__(self):
		printStr = ""
		printStr += "resultDir:\t" + str(self.resultDir) + "\n"
		printStr += "dataDir:\t" + str(self.dataDir) + "\n"
		printStr += "modelFile:\t" + str(self.modelFile) + "\n"
		printStr += "four2two weights:\t" + str(self.weights) + "\n"
		printStr += "numClass:\t" + str(self.numClass) + "\n"
		printStr += "model:\t" + str(self.model) + "\n"
		return printStr

	# Returns accuracy and loss on testData computed using 4-way classifier
	def test(self, testData, testLabel, batchSize=None):
	
		# Calculate number of batches using self.batchSize if batchSize given in this function is None
		batchSize = int(0.05*len(testData)) if batchSize is None else batchSize
		numBatches 	= math.ceil(len(testData)/batchSize)

		accuracy 	= 0.0
		totalLoss	= 0.0
		prediction  = [None]*len(testData)
		for batchNum in range(numBatches):

			sentences, targets, seqLens = prepBatch(batchSize, batchSize*batchNum, testData, testLabel, self.wordVocab)
			tagScores 	= self.model(sentences, seqLens)
			tagScores 	= tagScores.data.numpy() # Convert to numpy vectors
			
			for i in range(len(tagScores)):
				tagScores[i] =  np.exp(tagScores[i]) / np.sum(np.exp(tagScores[i]), axis=0) # Normalize using softmax
				
				probFake = 0.
				for j in range(len(tagScores[i])): # Convert probability distribution over 4 classes to over 2 class
					probFake += self.weights[j]*tagScores[i][j]

				if(probFake >= 0.5):
					prediction[batchNum*batchSize + i] = 0
				else:
					prediction[batchNum*batchSize + i] = 1

		
		totalPerClass 		= {c:0 for c in range(self.numClass)} # Total number of test points in class c
		retrievedPerClass 	= {c:0 for c in range(self.numClass)} # Number of test points labelled c
		truePerClass 		= {c:0 for c in range(self.numClass)} # True positives for each class 
		for i in range(len(testData)):
			if(prediction[i] == testLabel[i]):
				accuracy += 1
				truePerClass[prediction[i]] += 1

			retrievedPerClass[prediction[i]] += 1
			totalPerClass[testLabel[i]] += 1
		
		perClassResult = {}
		for c in range(self.numClass):
			precision 	= truePerClass[c]/retrievedPerClass[c] if retrievedPerClass[c] != 0 else 0
			recall 		= truePerClass[c]/totalPerClass[c] if truePerClass[c] != 0 else 0
			f1Score 	= 2*precision*recall/(precision + recall) if (precision + recall) != 0 else 0
			perClassResult[c] = (precision, recall, f1Score)

		accuracy = accuracy/len(testData)
	
		self.logger.info("Accuracy{:.3f}".format(accuracy))		
		for c in range(self.numClass):
			precision, recall ,f1 = perClassResult[c]
			self.logger.info("Class:{:d}\tPrecision\t{:.3f}".format(c,precision))
			self.logger.info("Class:{:d}\tRecall\t{:.3f}".format(c,recall))
			self.logger.info("Class:{:d}\tF1 Score\t{:.3f}".format(c,f1))
			
		return (accuracy, totalLoss, perClassResult)


if __name__ == '__main__':
	# torch.manual_seed(1)
	# np.random.seed(0);

	parser = argparse.ArgumentParser(description='Train LSTM model for fake news classification.')
	parser.add_argument('--dataDir', type=str, default=None, help='This folder contains training data. \
												It has files X_train.txt, y_train.txt, X_dev.txt, y_dev.txt, X_test.txt, y_test.txt')
	parser.add_argument('--resultDir', type=str,default=None, help='directory to store results in')
	parser.add_argument('--modelFile', type=str,default=None, help='saved 4-way classifier pickle file')
	args = parser.parse_args()

	paramDict  = {}
	paramDict["resultDir"]		= args.resultDir
	paramDict["dataDir"]   		= args.dataDir
	paramDict["modelFile"]  	= args.modelFile

	fourToTwo = FourToTwo(paramDict)
	fourToTwo.test(fourToTwo.testData, fourToTwo.testLabel)



	

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
from utils import prepBatch, read_X, read_y


class Trainer(object):
	"""docstring for Trainer"""

	trainData 	= None
	trainLabel 	= None
	devData 	= None
	devLabel 	= None
	testData	= None
	testLabel	= None
	def __init__(self, paramDict):
		super(Trainer, self).__init__()
		self.dataDir   		= paramDict["dataDir"]
		self.resultDir   	= paramDict["resultDir"]

		logging.basicConfig(filename=self.resultDir + "/logFile.txt",level=logging.DEBUG) 		# Get logger object
		# Add console to the logger, with level INFO so that INFO logs go to both console and file, while DEBUG logs go to file only
		self.logger = logging.getLogger()
		console = logging.StreamHandler()
		console.setLevel(logging.INFO)
		logging.getLogger('').addHandler(console)

		self.logger.info("Loading train,test and dev data...")
		self.loadData()

		self.logger.info("Creating vocabulary using train data...")
		# Create word vocabulary, which stores unique index for each word, and also create index2Word Mapping
		self.wordVocab, self.index2Word = self.createWordVocabulary(self.trainData)
		
		self.vocabSize 		= len(self.wordVocab.keys())
		self.trainEpoch		= paramDict["trainEpoch"]
		self.modelParams    = paramDict["modelParams"]
		self.modelParams["vocabSize"] = self.vocabSize;

		self.batchSize  	= self.modelParams["batchSize"]
		self.numClass		= self.modelParams["outputDim"]
		self.numBatches 	= math.ceil(len(self.trainData)/self.batchSize)

		self.logger.info("Initializing model parameters...")
		self.model 			= LSTM(**self.modelParams)

		self.logger.info("Loading pre-trained embeddings...")
		self.loadPreTrainedEmbeddings()
		self.logger.info("Model and other training details...")
		self.logger.info(str(self))


	def __str__(self):
		printStr = ""
		printStr += "dataDir:\t" 	+ str(self.dataDir) + "\n"
		printStr += "resultDir:\t" 	+ str(self.resultDir) + "\n"
		printStr += "VocabSize:\t" 	+ str(self.vocabSize) + "\n"
		printStr += "trainEpoch:\t" + str(self.trainEpoch) + "\n"		
		printStr += "batchSize:\t" 	+ str(self.batchSize) + "\n"
		printStr += "numBatches:\t" + str(self.numBatches) + "\n"

		printStr += "--"*20 + "\n" + str(self.model) + "\n" + "--"*20
		return printStr
		

	def sortData(self,data,label):

		sortedData = sorted(zip(data,label), key=lambda pair:len(pair[0]), reverse=True )
		data 	= [x for x,_ in sortedData]
		label 	= [x for _,x in sortedData]
		return data,label
		
	# Load train and validation data
	def loadData(self):

		self.trainData 	= read_X(self.dataDir + "/X_train.txt")
		self.trainLabel = read_y(self.dataDir + "/y_train.txt")
		self.trainData, self.trainLabel = self.sortData(self.trainData, self.trainLabel)
		
		self.devData 	= read_X(self.dataDir + "/X_dev.txt")
		self.devLabel 	= read_y(self.dataDir + "/y_dev.txt")
		self.trainData, self.trainLabel = self.sortData(self.devData, self.devLabel)

		self.testData 	= read_X(self.dataDir + "/X_test.txt")
		self.testLabel 	= read_y(self.dataDir + "/y_test.txt")
		self.trainData, self.trainLabel = self.sortData(self.testData, self.testLabel)

		self.trainSmallData 	= read_X(self.dataDir + "/X_1000.txt")
		self.trainSmallLabel 	= read_y(self.dataDir + "/y_1000.txt")
		self.trainData, self.trainLabel = self.sortData(self.trainSmallData, self.trainSmallLabel)

		# self.trainData = self.trainSmallData
		# self.trainLabel = self.trainSmallLabel

	def loadPreTrainedEmbeddings(self):
		word_vectors = KeyedVectors.load_word2vec_format('../../../dataset/GoogleNews-vectors-negative300.bin', binary=True)
		self.model.wordEmbeddings.weight.requires_grad=False
		for word in wordVocab:
			wordIdx = wordVocab[word]
			try:
				self.model.wordEmbeddings.weight[wordIdx] = torch.FloatTensor(word_vectors.get_vector(word))	
			except:
				self.logger.debug("Embedding not present in pre-trained {}".format(word))
		
	def train(self):

		self.logger.info("Training begins...")
		trainTime, trainLoss, trainAcc, perClassResult = {},{},{},{}
	
		# Train the model
		for e in range(self.trainEpoch):
			start = time.time()
			self.logger.info("Epoch No. : {}".format(e))
			totalLoss = 0
			for batchNum in range(self.numBatches):

				self.logger.info("Batch No. : {}".format(batchNum))
				sentences, targets, seqLens = prepBatch(self.batchSize, self.batchSize*batchNum, 
																self.trainData,self.trainLabel, self.wordVocab)

				self.model.zero_grad()
				tag_scores 	= self.model(sentences, seqLens) # Compute tag scores 
				loss 		= self.model.lossFunc(tag_scores, targets) # Compute loss 
				loss.backward()
				self.model.optimizer.step()
				totalLoss += loss.data

			end = time.time()
			trainTime[e] = end-start
			trainAcc[e], trainLoss[e],perClassResult[e]  = self.test(self.trainData, self.trainLabel)

		self.logger.info("Training ends...")

		self.logger.info("{}\t{}\t{}\t{}".format("Epoch", "Time", "Loss", "Acc"))
		for e in range(self.trainEpoch):
			self.logger.info("{:d}\t{:.3f}\t{:.3f}\t{:.3f}".format(e, trainTime[e], trainLoss[e], trainAcc[e]))
			
		self.logger.info("{}\t{}\t{}\t{}\t{}".format("Epoch","Class","Prec","Recl","F1"))
		for e in range(self.trainEpoch):
			for c in range(self.numClass):
				precision, recall ,f1 = perClassResult[e][c]
				self.logger.info("{:d}\t{:d}\t{:.3f}\t{:.3f}\t{:.3f}".format(e,c, precision, recall, f1))
			
	# Returns accuracy and loss on testData using current model
	def test(self, testData, testLabel, batchSize=None):
	
		# Calculate number of batches using self.batchSize if batchSize given in this function is None
		batchSize = self.batchSize if batchSize is None else batchSize
		numBatches 	= math.ceil(len(testData)/batchSize)

		accuracy 	= 0.0
		totalLoss	= 0.0
		prediction  = [None]*len(testData)
		for batchNum in range(numBatches):

			sentences, targets, seqLens = prepBatch(batchSize, batchSize*batchNum, testData, testLabel, self.wordVocab)
			tagScores 	= self.model(sentences, seqLens)
			loss 		= self.model.lossFunc(tagScores, targets)
			totalLoss 	+= loss.data[0]
			
			tagScores 	= tagScores.data.numpy() # Convert to numpy vectors
			targets   	= targets.data.numpy()

			for j in range(len(tagScores)):
				prediction[batchNum*batchSize + j] = int(np.argmax(tagScores[j]))
	
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
			recall 		= truePerClass[c]/totalPerClass[c] if totalPerClass[c] != 0 else 0
			f1Score 	= 2*precision*recall/(precision + recall) if (precision + recall) != 0 else 0
			perClassResult[c] = (precision, recall, f1Score)

		accuracy = accuracy/len(testData)
		return (accuracy, totalLoss, perClassResult)

	# Create dictionary for the vocabulary. Vocabulary is a dictionary with word as keys and value as a unique index
	# Returns wordVocabulary and index2Word mappings as well
	def createWordVocabulary(self, corpus):

		wordVocab = {}
		for sentence in corpus:
			for word in sentence:
				wordVocab[word] = 1

		wordVocab["OOV"] = 1 # Add OOV word to vocabulary
		wordVocab = {w:ctr for ctr,w in enumerate(wordVocab.keys())} # Assign index to each word in vocabulary
		index2Word  = {wordVocab[w]:w for w in wordVocab} # Create index to word mappings

		return wordVocab, index2Word

	def saveModel(self,filename=None):
		if filename is None:
			filename = self.resultDir + "/model.pkl"
		else:
			filename = self.resultDir + "/" + filename

		with open(filename,"wb") as f:
			dataToPickle = (self.model, self.wordVocab, self.index2Word)
			pickle.dump(dataToPickle,f )

if __name__ == '__main__':
	torch.manual_seed(1)
	np.random.seed(0);

	parser = argparse.ArgumentParser(description='Train LSTM model for fake news classification.')
	parser.add_argument('--dataDir', type=str, default=None, help='This folder contains training data. \
												It has files X_train.txt, y_train.txt, X_dev.txt, y_dev.txt, X_test.txt, y_test.txt')
	parser.add_argument('--resultDir', type=str,default=None, help='directory to store results in')
	parser.add_argument('--trainEpoch', type=int, default=10, help='number of epochs to run on training data.')
	parser.add_argument('--numClass', type=int, default=4, help='number of classed in training data.')
	args = parser.parse_args()

	paramDict  = {}
	paramDict["resultDir"]		= args.resultDir
	paramDict["dataDir"]   		= args.dataDir
	paramDict["trainEpoch"]		= args.trainEpoch
	paramDict["modelParams"]    = {"hiddenDim":10, "embedDim":10, "batchSize":100, "outputDim":args.numClass}

	trainer = Trainer(paramDict)
	trainer.train()
	trainer.test(trainer.trainSmallData,trainer.trainSmallLabel)
	trainer.saveModel("test.pkl")

# This file is for extracting data and cleaning it up


import sys
import os
from nltk.tokenize import RegexpTokenizer
import numpy as np

from gensim.models import KeyedVectors

word_vectors = KeyedVectors.load_word2vec_format('../../dataset/GoogleNews-vectors-negative300.bin', binary=True) 

print ("here")

print (word_vectors.get_vector('dog'))
#print (len(word_vectors))
#print (word_vectors['dog'])
#print (len(word_vectors['dog']))

'''X = []
y = []

filepath = '../../dataset/newsfiles/xtrain.txt'  
with open(filepath) as fp:  
	line = fp.readline()

	while line:
		
		y.append(int(line[0]) - 1)
		temp = line[2:]

		tokenizer = RegexpTokenizer(r'\w+')
		temp = tokenizer.tokenize(temp)

		X.append(temp)

		line = fp.readline()


np.save('X_train.npy', X)
np.save('y_train.npy', y)'''


'''train_los = np.load('train_los.npy')
train_acc = np.load('train_acc.npy')
valid_acc = np.load('valid_acc.npy')
valid_los = np.load('valid_los.npy')


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
plt.show()'''

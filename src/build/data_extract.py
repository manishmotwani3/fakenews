# This file is for extracting data and cleaning it up


import sys
import os
from nltk.tokenize import RegexpTokenizer
import numpy as np

#from gensim.models import KeyedVectors

#word_vectors = KeyedVectors.load_word2vec_format('dataset/GoogleNews-vectors-negative300.bin', binary=True) 

#print ("here")

#print (get_vector("entity"))
#print (len(get_vector("entity")))

X = []
y = []

filepath = 'dataset/newsfiles/xdev.txt'  
with open(filepath) as fp:  
	line = fp.readline()

	while line:
		
		y.append(int(line[0]))
		temp = line[2:]

		tokenizer = RegexpTokenizer(r'\w+')
		temp = tokenizer.tokenize(temp)

		X.append(temp)

		line = fp.readline()


np.save('X_valid.npy', X)
np.save('y_valid.npy', y)
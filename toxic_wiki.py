#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  6 17:21:43 2019

@author: cthomasbrittain
"""
# Data are hosted by Kaggle
# https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data
# The Word Embeddings are created Google.
# https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit
# pip install gensim
# https://radimrehurek.com/gensim/models/keyedvectors.html
# Also install nltk
# sudo pip install -U nltk (Natural Language Toolkit)
# https://www.depends-on-the-definition.com/guide-to-word-vectors-with-gensim-and-keras/
# https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html
# https://github.com/keras-team/keras/blob/master/examples/pretrained_word_embeddings.py
from __future__ import print_function

import os
import sys
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, GlobalMaxPooling1D
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model
from keras.initializers import Constant
import gensim.downloader as api
import gensim
import pandas as pd

############################
# Convenience Macros
############################

BASE_DIR = '/Users/cthomasbrittain/dl-nlp/data/'
GLOVE_DIR = os.path.join(BASE_DIR)
GOOGLE_NEWS_PATH = BASE_DIR + 'GoogleNews-vectors-negative300.bin'
TRAIN_TEXT_DATA_DIR = BASE_DIR + 'toxic-comment/test.csv'
TRAIN_TEXT_LABELS_DIR = BASE_DIR + 'toxic-comment/test_labels.csv'
MAX_SEQUENCE_LENGTH = 100
MAX_NUM_WORDS = 20000
EMBEDDING_DIM = 25
VALIDATION_SPLIT = 0.2

##############################
# Load GoogleNews Embeddings
##############################

print('Loading word vectors.')

# Load embeddings
model = gensim.models.KeyedVectors.load_word2vec_format(GOOGLE_NEWS_PATH, binary=True)
#info = api.info()                       # show info about available models/datasets
model = api.load("glove-twitter-25")    # download the model and return as object ready for use

vocab_size = len(model.vocab)
embeddings = model.get_keras_embedding()

index2word = model.index2entity
word2index = {}
for index in range(vocab_size):
    word2index[model.index2word[index]] = index


############################################
# Get labels
############################################

print('Preparing embedding matrix.')
labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
with open(TRAIN_TEXT_LABELS_DIR) as f:
    y = pd.read_csv(TRAIN_TEXT_LABELS_DIR)[labels].values
    
############################################
# Convert Toxic Comments to Word Vectors
############################################

print('Processing text dataset')

with open(TRAIN_TEXT_LABELS_DIR) as f:    
    toxic_comments = pd.read_csv(TRAIN_TEXT_DATA_DIR)

comment_vectors = []
with open(TRAIN_TEXT_DATA_DIR) as f:
    for index, text in enumerate(toxic_comments['comment_text'].tolist()):
        words = text.split()
        word_vec = []
        for word in words:
            try:
                word_vec.append(word2index[word])
            except:
                continue
        comment_vectors.append(word_vec)

print('Found %s comments.' % len(comment_vectors))


############################################
# Pad Sequences
############################################

data = pad_sequences(comment_vectors, maxlen=MAX_SEQUENCE_LENGTH)


############################################
# Convert Toxic Comment Vectors to Embedding Layer
############################################

#for index, row in enumerate(data):
#    embedding_matrix[index] = model.get_vector()

embedding_matrix = np.zeros((len(word2index) + 1, EMBEDDING_DIM))
for word, i in word2index.items():
    embedding_vector = model.get_vector(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

# load pre-trained word embeddings into an Embedding layer
# note that we set trainable = False so as to keep the embeddings fixed
embedding_layer = Embedding(len(word2index),
                            EMBEDDING_DIM,
                            embeddings_initializer=Constant(embedding_matrix),
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)

############################################
# Train
############################################

print('Training model.')

# train a 1D convnet with global maxpooling
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embeddings(sequence_input)
x = Conv1D(128, 5, activation='relu')(embedded_sequences)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 3, activation='relu')(x)
x = GlobalMaxPooling1D()(x)
x = Dense(128, activation='relu')(x)
preds = Dense(len(labels), activation='softmax')(x)

model = Model(sequence_input, preds)
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

model.fit(data, y,
          batch_size=128,
          epochs=10)

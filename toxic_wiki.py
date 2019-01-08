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

import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, GlobalMaxPooling1D, Conv1D, Embedding, MaxPooling1D
from keras.models import Model
from keras.initializers import Constant
import gensim.downloader as api
import pandas as pd

############################
# Convenience Macros
############################

BASE_DIR = '/Users/cthomasbrittain/dl-nlp/data/'
TRAIN_TEXT_DATA_DIR = BASE_DIR + 'toxic-comment/train.csv'
MAX_SEQUENCE_LENGTH = 100
MAX_NUM_WORDS = 20000
EMBEDDING_DIM = 25
VALIDATION_SPLIT = 0.0

##############################
# Load GoogleNews Embeddings
##############################

print('Loading word vectors.')

# Load embeddings
info = api.info()                       # show info about available models/datasets
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
    
with open(TRAIN_TEXT_DATA_DIR) as f:
    toxic_comments = pd.read_csv(TRAIN_TEXT_DATA_DIR)

print('Getting Comment Labels.')
prediction_labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
with open(TRAIN_TEXT_DATA_DIR) as f:
    labels = toxic_comments[prediction_labels].values
    
############################################
# Convert Toxic Comments to Sequences
############################################

print('Processing text dataset')

tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer.fit_on_texts(toxic_comments['comment_text'].fillna("DUMMY_VALUE").values)
sequences = tokenizer.texts_to_sequences(toxic_comments['comment_text'].fillna("DUMMY_VALUE").values)

word_index = tokenizer.word_index

print('Found %s sequences.' % len(sequences))


############################################
# Pad Sequences
############################################
data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)


###################################################
# Convert Toxic Comment Vectors to Embedding Layer
###################################################
num_words = min(MAX_NUM_WORDS, len(word_index)) + 1
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word_index.items():
    try:
        embedding_vector = model.get_vector(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    except:
        continue

# Load pre-trained word embeddings into an Embedding layer.
# Note, that we set trainable = False so as to keep the embeddings fixed.
embedding_layer = Embedding(len(word2index),
                            EMBEDDING_DIM,
                            embeddings_initializer=Constant(embedding_matrix),
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)

###################################################
# Test / Train Split
###################################################
nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])
x_train = data[:-nb_validation_samples]
y_train = labels[:-nb_validation_samples]
x_val = data[-nb_validation_samples:]
y_val = labels[-nb_validation_samples:]


############################################
# Train
############################################

print('Building model...')
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
x = Conv1D(128, 3, activation='relu')(embedded_sequences)
x = MaxPooling1D(3)(x)
x = Conv1D(128, 3, activation='relu')(x)
x = MaxPooling1D(3)(x)
x = Conv1D(128, 3, activation='relu')(x)
x = MaxPooling1D(3)(x)  # global max pooling
x = GlobalMaxPooling1D()(x)
x = Dense(128, activation='relu')(x)
preds = Dense(y_train.shape[1], activation='softmax')(x)

model = Model(sequence_input, preds)
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])


print('Training model.')
# happy learning!
history = model.fit(x_train, y_train, validation_data=(x_val, y_val),
          epochs=2, batch_size=128)


############################################
# Predict
############################################

def prepare_sample(sequence, tokenizer, max_length):
    sequence = tokenizer.texts_to_sequences(sequence)
    data = pad_sequences(sequence, maxlen=max_length)
    return data


# Create a test sequence
sequence = [("""
                COCKSUCKER BEFORE YOU PISS AROUND ON MY WORK
        """)]

# Convert the sequence to tokens and pad it.
sequence = prepare_sample(sequence, tokenizer, MAX_SEQUENCE_LENGTH)

# Make a prediction
sequence_prediction = model.predict(sequence, batch_size=None, verbose=1, steps=None)

# Take only the first of the batch of predictions
sequence_prediction = pd.DataFrame(sequence_prediction[0:1,:])

# Label the predictions
sequence_prediction.columns = prediction_labels

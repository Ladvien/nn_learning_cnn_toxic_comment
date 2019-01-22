#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 05:19:35 2019

@author: cthomasbrittain
"""

##############################
# Load GoogleNews Embeddings
##############################
import gensim.downloader as api
print('Loading word vectors.')

# Load embeddings
info = api.info()                       # show info about available models/datasets
embedding_model = api.load("glove-wiki-gigaword-300")    # download the model and return as object ready for use

vocab_size = len(embedding_model.vocab)

index2word = embedding_model.index2word
word2idx = {}
for index in range(vocab_size):
    word2idx[embedding_model.index2word[index]] = index

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 16:57:47 2019

@author: shayan
"""

# In[]:

import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation
from keras.layers.embeddings import Embedding
# In[]:A function is defined to read the text, removing the hyperlinks in the text and returning in the form of lists of label and text

def read(train):
    text=[]
    label=[]
    for i,t in enumerate(train):
        t = t.strip(" ") 
        t = re.sub(r'http\S+', '', t)     
        t=word_tokenize(t)
        text.append(t[1:])
        try:
            if t[0] =='No':
                label.append(0)
            else:
                label.append(1)
        except:
            continue
    return text, label

# In[]:Lemmatization of text usimg WordNetLemmatizer  
def preprocess(text):
    stop = set(stopwords.words('english')) 
    pre_text = []
    for tokens in text:
        lemmatizer = WordNetLemmatizer()
        lst = " ".join(lemmatizer.lemmatize(token) for token in tokens if token not in stop and token.isalnum())
        pre_text.append(lst)
    return pre_text
# In[]: Producing the embedding matrix representing all the words as glove vector (300 dimensional vector)

def embedding(x_train, x_test):
    t = Tokenizer()
    t.fit_on_texts(x_train)
    embeddings_index = dict()
    vocab_size = len(t.word_index) + 1
    encoded_docs_train = t.texts_to_sequences(x_train)
    encoded_docs_test = t.texts_to_sequences(x_test)
    max_length = 20
    padded_docs_train = pad_sequences(encoded_docs_train, maxlen=max_length, padding='post')
    padded_docs_test = pad_sequences(encoded_docs_test, maxlen=max_length, padding='post')
    
    f = open('glove.6B.300d.txt')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    
    print('Loaded %s word vectors.' % len(embeddings_index))
    # create a weight matrix for words in training docs
    embedding_matrix = np.zeros((vocab_size, 300))
    for word, i in t.word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            
    return vocab_size, embedding_matrix, padded_docs_train, padded_docs_test

# In[]: Loading the train and test files located in the same folder only
    
train=open("train.txt","r")
test=open("test.txt","r")

train=train.read().split("\n")[:-1]
test=test.read().split("\n")[:-1]

x_train=[]
train_label=[]
x_test=[]
test_label=[]

x_train, train_label=read(train)
x_test, test_label=read(test)

x_train=preprocess(x_train)
x_test=preprocess(x_test)

vocab_size, embedding_matrix, x_train, x_test= embedding(x_train, x_test)
# In[]: Defining a simple neural network model on the embedding matrix obtained as input to classify the text

model = Sequential()
e = Embedding(vocab_size, 300, weights=[embedding_matrix], input_length=20, trainable=True)
model.add(e)
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
# compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
# summarize the model
print(model.summary())
# fit the model
model.fit(x_train, train_label, epochs=50, verbose=0)
# evaluate the model
train_loss, train_accuracy = model.evaluate(x_train, train_label, verbose=0)
print('Training Accuracy: %f' % (train_accuracy*100))
# evaluate the model
test_loss, test_accuracy = model.evaluate(x_test, test_label, verbose=0)
print('Test Accuracy: %f' % (test_accuracy*100))


# In[]: Defining a dense model containing convolution layers with dropouts and max-pooling involved and finally a LSTM layer classifying text into two categories
vocabulary_size=vocab_size
model_conv = Sequential()
model_conv.add(Embedding(vocabulary_size, 300, weights=[embedding_matrix], input_length=20, trainable=False))
model_conv.add(Dropout(0.3))
model_conv.add(Conv1D(64, 5, activation='relu'))
model_conv.add(Dropout(0.3))
model_conv.add(Conv1D(32, 4, padding='valid', activation='relu'))
model_conv.add(MaxPooling1D(pool_size=4))
model_conv.add(Dropout(0.3))
model_conv.add(LSTM(50))
model.add(Dense(10,activation='sigmoid'))
model_conv.add(Dense(1, activation='sigmoid'))
# compile the model
model_conv.compile(loss='binary_crossentropy', optimizer='adam',    metrics=['accuracy'])
#summarize the model
print(model_conv.summary())
#fit the model
model_conv.fit(x_train, train_label, validation_split=0.2, epochs = 4)
# evaluate the model
train_loss1, train_accuracy1 = model_conv.evaluate(x_train, train_label, verbose=0)
print('Training Accuracy: %f' % (train_accuracy1*100))
# evaluate the model
test_loss1, test_accuracy1 = model_conv.evaluate(x_test, test_label, verbose=0)
print('Test Accuracy: %f' % (test_accuracy1*100))


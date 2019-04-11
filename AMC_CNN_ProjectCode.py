#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 21:25:01 2019

@author: amith
"""

import os,random
import numpy as np
import theano as th
import theano.tensor as T
from keras.utils import np_utils
import keras.models as models
from keras.layers.core import Reshape,Dense,Dropout,Activation,Flatten
from keras.layers.noise import GaussianNoise
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.regularizers import *
from keras.optimizers import adam
import matplotlib.pyplot as plt
import seaborn as sns
import pickle,random,sys,keras # for unpickling



Xd = pickle.load(open("RML2016.10a_dict.pkl",'rb'),encoding='latin')
snrs,mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1,0])
X = []  
lbl = []
for mod in mods:
    for snr in snrs:
        X.append(Xd[(mod,snr)])
        for i in range(Xd[(mod,snr)].shape[0]):  lbl.append((mod,snr))
X = np.vstack(X)


""" Splitting the Dataset into Test Set and Training Set """
np.random.seed(2016) # A seed to make random numbers predictable
n_examples = X.shape[0] 
n_train = 110000 # 110000
train_idx = np.random.choice(range(0,n_examples), size=n_train, replace=False) 
test_idx = list(set(range(0,n_examples)))
X_train= X[train_idx] # 1,10,000 samples in training set
X_test = X[test_idx] # 2,20,000 samples in test set
 # to one hot encoder since the outputs are modulation types in text format
def __len__(k):
    return len(k)
def to_onehot(yy):
    yy1 = np.zeros([__len__(yy),max(yy)+1])
    yy1[np.arange(__len__(yy)),yy] =1
    return yy1

one_hot1=[]
one_hot2=[]
for x in train_idx:
    one_hot1= np.append(one_hot1,mods.index(lbl[x][0]))

    
for y in test_idx:
    one_hot2= np.append(one_hot2,mods.index(lbl[y][0]))

#one_hot1=map(lambda x: mods.index(lbl[x][0]),train_idx)
#one_hot2=map(lambda x: mods.index(lbl[x][0]),test_idx)
    
one_hot1int = one_hot1.astype(int)
one_hot2int = one_hot2.astype(int)    
Y_train = to_onehot(one_hot1int)
Y_test = to_onehot(one_hot2int)

#Y_train
# Data set split into training and test cases
# training is done by random samples from the dataset with a total of 110000 vectors 
# and is tested on all test cases in dataset

in_shp = list((X_train.shape[1:]))
print (X_train.shape,in_shp)
classes = mods

# BUILDING THE NEURAL NETWORK ARCHITECTURE
drop_out = 0.5
model = models.Sequential() # from keras library
# INPUT LAYER 2 x 128
model.add(Reshape([1]+in_shp,input_shape=in_shp))
model.add(ZeroPadding2D((0,2)))
# ZERO PADDING required to make output size equal to input size
model.add(Convolution2D(256,1,3,border_mode='valid',activation="relu",name="conv1",init='glorot_uniform'))
model.add(Dropout(drop_out))
# Dropout is required to prevent overfitting 

#LAYER 2
model.add(ZeroPadding2D((0,2)))
model.add(Convolution2D(80,2,3,border_mode="valid",activation="relu",name="conv2",init = 'glorot_uniform'))
model.add(Dropout(drop_out))

model.add(Flatten())
#DENSE LAYER OR FULLY CONNECTED LAYER
model.add(Dense(256,activation='relu',init='he_normal',name="dense1"))
model.add(Dropout(drop_out))
# 2nd dense layer
model.add(Dense(len(classes),init='he_normal',name="dense2"))
model.add(Activation('softmax'))
model.add(Reshape([len(classes)]))

model.compile(loss='categorical_crossentropy',optimizer='adam')
model.summary()

nb_epoch = 100
batch_size = 1024

filepath='CNN_AMC.wts.h5'

history =model.fit(X_train,
                   Y_train,
                   batch_size=batch_size,
                   nb_epoch=nb_epoch,
                   show_accuracy=False,
                   verbose=2,
                   validation_data=(X_test,Y_test),
                   callbacks=[keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, mode='auto'),
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='auto')])

model.load_weights(filepath)







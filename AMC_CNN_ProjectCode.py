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
import pickle,sys,keras # for unpickling
import tkinter as tk
root = tk.Tk()

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

in_shp = list((X_train.shape[1:])) #[2,128]
print (X_train.shape,in_shp) 
classes = mods

# BUILDING THE NEURAL NETWORK ARCHITECTURE
drop_out = 0.5
model = models.Sequential() # from keras library
# INPUT LAYER 2 x 128
# Reshape(self,target_shape,**kwargs)
model.add(Reshape([1]+in_shp,input_shape=in_shp)) # no change in shape in this command
model.add(ZeroPadding2D((0,1)))
# ZERO PADDING required to make output size equal to input size

model.add(Convolution2D(256,kernel_size=(3,1),strides=(1,1),border_mode='valid',activation="relu",name="conv1",init='glorot_uniform',data_format='channels_first'))
model.add(Dropout(drop_out))
# Dropout is required to prevent overfitting 

#LAYER 2
model.add(ZeroPadding2D((0,2)))

model.add(Convolution2D(80,kernel_size=(3,2),strides=(1,1),border_mode="valid",activation="relu",name="conv2",init = 'glorot_uniform',data_format='channels_first'))
model.add(Dropout(drop_out))
#
model.add(Flatten())
#DENSE LAYER OR FULLY CONNECTED LAYER
model.add(Dense(256,activation='relu',init='he_normal',name="dense1"))
model.add(Dropout(drop_out))
# 2nd dense layer
model.add(Dense(len(classes),init='he_normal',name="dense2"))
model.add(Activation('softmax'))

model.add(Reshape([len(classes)]))

model.compile(loss='categorical_crossentropy',optimizer='adam')
# ADAM optimizer provides gradient normalization and momentum which reduces the importance of
#learning rate.
model.summary()

nb_epoch = 100
batch_size = 512 #1024

filepath='CNN_AMC.wts.h5'


""" callbacks is a set of functions to be applied at given stages during training
   callbacks.Modelcheckpoint is used to save the model after every epoch . The model checkpoints will be saved with epoch
   number and the validation loss in the filename """ 
history =model.fit(X_train,
                   Y_train,
                   batch_size=batch_size,
                   nb_epoch=nb_epoch,
                   verbose=1,
                   validation_data=(X_test,Y_test),
                   callbacks=[keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, mode='auto'),
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='auto')])

# reloading the best weights after the training is finished.
model.load_weights(filepath) # returns the trained tensors as list of numpy arrays

# model is trained, accuracy is shown ,loss curves are plotted, confusion matrix are to be created.
plt.figure()
plt.title('Training performance')
plt.plot(history.epoch, history.history['loss'], label='train loss+error')
plt.plot(history.epoch, history.history['val_loss'], label='val_error')
plt.legend()


def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues, labels=[]):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    
X_test.shape[0]    
test_Y_hat = model.predict(X_test, batch_size=batch_size)
conf = np.zeros([len(classes),len(classes)])
confnorm = np.zeros([len(classes),len(classes)])
for i in range(0,X_test.shape[0]):
    j = list(Y_test[i,:]).index(1)
    k = int(np.argmax(test_Y_hat[i,:]))
    conf[j,k] = conf[j,k] + 1
for i in range(0,len(classes)):
    confnorm[i,:] = conf[i,:] / np.sum(conf[i,:])
plot_confusion_matrix(confnorm, labels=classes) 
plt.savefig('confusion_matrix_all_SNR',dpi=400)
cor = np.sum(np.diag(conf))
ncor = np.sum(conf) - cor
acc_all = 1.0*cor/(cor+ncor)

acc = {}


test_SNRs=[]
for i in test_idx:
    test_SNRs = np.append(test_SNRs,lbl[i][1])

for snr in snrs:
    #test_SNRs = map(lambda x: lbl[x][1], test_idx)
    test_X_i = X_test[np.where(np.array(test_SNRs)==16)]
    test_Y_i = Y_test[np.where(np.array(test_SNRs)==16)]    

    # estimate classes
    test_Y_i_hat = model.predict(test_X_i)
    conf = np.zeros([len(classes),len(classes)])
    confnorm = np.zeros([len(classes),len(classes)])
    for i in range(0,test_X_i.shape[0]):
        j = list(test_Y_i[i,:]).index(1)
        k = int(np.argmax(test_Y_i_hat[i,:]))
        conf[j,k] = conf[j,k] + 1
    for i in range(0,len(classes)):
        confnorm[i,:] = conf[i,:] / np.sum(conf[i,:])
    plt.figure()
    plot_confusion_matrix(confnorm, labels=classes, title="ConvNet Confusion")
    plt.savefig("16SNR",dpi=400)
    cor = np.sum(np.diag(conf))
    ncor = np.sum(conf) - cor
    acc[snr] = 1.0*cor/(cor+ncor)

#mapped=map(lambda x: acc[x], snrs)
#acc_list = list(acc.values())

lists = sorted(acc.items()) # sorted by key, return a list of tuples

x, y = zip(*lists)   
plt.plot(x,y)
plt.xlabel("Signal to Noise Ratio")
plt.ylabel("Classification Accuracy")
plt.grid(True)
plt.title("CNN on RadioML 2016.10 Alpha")
plt.savefig('Accuracy_SNR',dpi=400)

fd = open('results_cnn2_d0.5.dat','wb')
cPickle.dump( ("CNN2", 0.5, acc) , fd )

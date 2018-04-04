#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 31 15:50:43 2018

@author: dykuang

Using a VGG-16 perceptual loss for directing the spatial deformer
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage.io import imread
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D,  UpSampling2D, multiply, \
                            concatenate, Flatten, Dense, Lambda
                            
from keras.optimizers import Adam, SGD
import keras.backend as K
from spatial_deformer_net import SpatialDeformer
from keras.losses import mean_squared_error


#from skimage import transform as tsf
#tform = tsf.SimilarityTransform(scale=1.0, rotation=0, translation=(5, 5))

#------------------------------------------------------------------------------
# Hyperparamters/Global setting
#------------------------------------------------------------------------------
epochs = 4
batch_size = 8
res = 224
input_shape_G = (res,res,2)

#------------------------------------------------------------------------------
# Data Preparation
#------------------------------------------------------------------------------
data_path = r'datasets/oasis/'

import os 
train = []
for temp in os.listdir(data_path):
     brain = imread(os.path.join(data_path, temp), as_grey = True)
     brain = resize(brain, (res,res), mode='reflect') #smooth before resample?
     train.append(brain)

train = np.array(train)
train = train.astype('float32')


#stack any two templates, forming training set
from itertools import combinations
x_train = np.zeros([380,res,res,2])
for i, ind in enumerate(combinations(range(20), 2)):
     x_train[i,:,:,0] = train[ind[0]]
     x_train[i,:,:,1] = train[ind[1]] 
     
     x_train[i+190,:,:,0] = train[ind[1]]
     x_train[i+190,:,:,1] = train[ind[0]]
     
y_train = np.expand_dims(x_train[:,:,:,1],3)

sample_choice = np.random.choice(380, 9, replace = False)
def vis(G, choice = sample_choice):
    plt.figure()
    
    sample = G.predict(x_train[choice])
    for i in range(3):
        for j in range(3):
            plt.subplot(3, 3, 3*i+j+1)
            plt.imshow(sample[3*i+j,:,:,0])
            plt.axis('off')

def vis_target(choice = sample_choice):
    sample = y_train[choice]
    plt.figure()
    for i in range(3):
        for j in range(3):
            plt.subplot(3, 3, 3*i+j+1)
            plt.imshow(sample[3*i+j,:,:,0])
            plt.axis('off')
    plt.suptitle('target images')


#cat1 = imread('cat4.jpg', as_grey = True)
#cat2 = imread('cat5.jpg', as_grey = True)
#cat1 = resize(cat1, (res,res), mode='reflect')
#cat2 = resize(cat2, (res,res), mode='reflect')
#x_train[0,:,:,0] = cat1
#x_train[0,:,:,1] = cat2
#
#y_train[0,:,:,0] = cat2 
    
#x_train[0,:,:,0] = tsf.warp(x_train[0,:,:,0], tform)
#------------------------------------------------------------------------------
# Some util functions
#------------------------------------------------------------------------------
def set_trainability(model, flag = False): # need to call compile() after this?
    model.trainable = flag
    for layer in model.layers:
        layer.trainable = flag
        

#------------------------------------------------------------------------------
# SDN part
#------------------------------------------------------------------------------
#from spatial_transformer_net import SpatialTransformer
def SDN(inputs):
    
    zz = Conv2D(64, (3,3), padding = 'same')(inputs)
#    zzz = Conv2D(64, (3,3), padding = 'same')(zz)
    
    zzz = MaxPooling2D((2,2))(zz)
    zzz = Conv2D(128, (3,3), padding = 'same')(zzz)
    
    zzz = UpSampling2D((2,2))(zzz) 
    zzz = Conv2D(64, (3,3), padding = 'same')(zzz)

    
    zzzz = multiply([zz, zzz]) 
    zzzz = Conv2D(2, (3,3), padding = 'same',
                      kernel_initializer= 'zeros',
                      bias_initializer = 'zeros',
                      activation = 'linear')(zzzz)
    
    locnet = Model(inputs, zzzz)
     
    x1 = SpatialDeformer(localization_net=locnet,
                             output_size=(input_shape_G[0],input_shape_G[1]), 
                             input_shape=input_shape_G)(inputs)
#    x1 = SpatialTransformer(localization_net=locnet,
#                             output_size=(input_shape_G[0],input_shape_G[1]), 
#                             input_shape=input_shape_G)(inputs)
    
    return x1

SDN_in = Input(shape = input_shape_G)
SDN_out = SDN(SDN_in)

sdn = Model(SDN_in, SDN_out)


#------------------------------------------------------------------------------
# VGG part
#------------------------------------------------------------------------------


from keras.applications.vgg16 import VGG16
#from keras.preprocessing import image
#from keras.applications.vgg16 import preprocess_input

# a connection part from SDN to VGG
VGG_mean = np.array([123.68, 116.779, 103.939]).reshape((1,1,1,3))
def preprocess(x):
#    xxx = Lambda(lambda x: x[:,:,:,:1])(x)
    xxx = K.concatenate([x,x,x])
    xxx = xxx - VGG_mean/255
    return xxx

XtoVGG = Lambda(preprocess)(SDN_out)
#XtoVGG = concatenate([SDN_out*255-123.68, SDN_out*255-116.779, SDN_out*255-103.939])     

#weights = [1,0,0,1,0]
#selected_layers = ['block1_conv2', 'block2_conv2', 
#                   'block3_conv3', 'block4_conv3',
#                   'block5_conv3']
selected_layers = ['block4_conv3']

base_model = VGG16(weights='imagenet', include_top=False)
set_trainability(base_model, False)

selected_output = [base_model.get_layer(L).output
                   for L in selected_layers]
loss_model = Model(base_model.input, selected_output)
set_trainability(base_model, False)

x_output = loss_model(XtoVGG)
whole_model = Model(SDN_in, x_output)

Y_train = np.concatenate([y_train-123.68/255, y_train-116.779/255, y_train-103.939/255], axis = -1)
Y_loss = loss_model.predict(Y_train[:1])

# custom loss function will apply to EACH output, NOT the whole output
#def loss_perceptual(y_True, y_Pred):
#    loss = 0
#    for i in range(5):
#        loss+=weights[i]*K.mean(K.pow(y_True[2]-y_Pred[2],2))
#        
#    return loss

from keras.layers import dot
def corr(y_True, y_Pred):
    h = K.shape(y_True)[1]
    w = K.shape(y_True)[2]
    d = K.shape(y_True)[3]
    
    y_True = K.reshape(y_True, (-1, h*w, d))
    y_Pred = K.reshape(y_Pred, (-1, h*w, d))
    
    cc = dot([y_Pred, y_Pred], 2, True) # use batch_dot? permuation first? same.
    
#    cc = K.relu(cc)

    ccT = dot([y_True, y_True], 2, True) # how to initialize eye properly?
    
    return K.mean(K.pow(cc-ccT , 2))

whole_model.compile(loss = corr, 
#                    loss_weights= weights,
                    optimizer='adam')

# pretrain with sdn itself?
#from BrainSDN import customLoss
#sdn.compile(loss = customLoss, 
#              optimizer = Adam(decay=1e-5),
#              )   
#sdn.fit(x_train[:1], y_train[:1], epochs=epochs)

whole_model.fit(x_train[:1], Y_loss, epochs=50)

aa = sdn.predict(x_train[:1])
plt.figure()
plt.subplot(1,2,1)
plt.imshow(aa[0,:,:,0], cmap='gray') 
plt.subplot(1,2,2)
plt.imshow(x_train[0,:,:,1], cmap='gray')






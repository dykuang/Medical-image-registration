#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 31 15:50:43 2018

@author: dykuang

Using a VGG-19 perceptual loss for directing the spatial deformer
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
def SDN(inputs):
    
    zz = Conv2D(64, (3,3), padding = 'same')(inputs)
    zzz = Conv2D(64, (3,3), padding = 'same')(zz)
    
    zzz = MaxPooling2D((2,2))(zzz)
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
    
    return x1

SDN_in = Input(shape = input_shape_G)
SDN_out = SDN(SDN_in)

sdn = Model(SDN_in, SDN_out)


#------------------------------------------------------------------------------
# VGG part
#------------------------------------------------------------------------------

"""
TODO: 
    preprocess?
    define a loss function on top !! something like content/style loss?
"""


from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input

# a connection part from SDN to VGG
VGG_mean = np.array([123.68, 116.779, 103.939]).reshape((1,1,1,3))
def preprocess(x):
    xxx = K.concatenate([x,x,x])
    xxx = xxx*255 - VGG_mean
    return xxx

XtoVGG = Lambda(preprocess)(SDN_out)
#XtoVGG = concatenate([SDN_out*255-123.68, SDN_out*255-116.779, SDN_out*255-103.939])     

weights = [1,1,1,1,1]
selected_layers = ['block1_conv2', 'block2_conv2', 
                   'block3_conv3', 'block4_conv3',
                   'block5_conv3']

base_model = VGG16(weights='imagenet', include_top=False)
set_trainability(base_model, False)

selected_output = [base_model.get_layer(selected_layers[i]).output
                   for i in range(5)]
loss_model = Model(base_model.input, selected_output)
set_trainability(base_model, False)

x_output = loss_model(XtoVGG)
whole_model = Model(SDN_in, x_output)

Y_train = np.concatenate([y_train*255-123.68, y_train*255-116.779, y_train*255-103.939], axis = -1)
#Y_loss = loss_model.predict(Y_train)

def loss_perceptual(y_True, y_Pred):
    loss = 0
    for j in range(5):
        loss += weights[j]*K.mean(mean_squared_error(y_True[j], y_Pred[j]))
        
    return loss

whole_model.compile(loss = 'mse',
                    optimizer=SGD())


whole_model.fit(x_train[:3], loss_model.predict(Y_train[:3]))

 







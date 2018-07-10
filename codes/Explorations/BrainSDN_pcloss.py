# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 10:20:19 2018

@author: dykua

This script test if using perceptual loss will be better in registration

* Train sdn
* Train a discriminator/a pretrained one maybe
* Train sdn again with perceptual loss

"""

import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage.io import imread
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D,  UpSampling2D, multiply, \
                            concatenate, Flatten, Dense
from keras.optimizers import Adam
import keras.backend as K
from spatial_deformer_net import SpatialDeformer

#------------------------------------------------------------------------------
# Hyperparamters/Global setting
#------------------------------------------------------------------------------
epochs = 4
batch_size = 8
res = 64
input_shape_G = (res,res,2)
input_shape_D = (res,res,1)

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
        
def TF_blend(G, batch_size):
    num = int(batch_size/2)
    choice = np.random.choice(380, num, replace = False) # repeated sampels when train?
    XT = y_train[choice,:]     
    XF = G.predict(x_train[choice,:])
    blend = np.concatenate((XT, XF),axis = 0)
    
    label = np.zeros((batch_size, 2))
    label[:num, 1] = 1
    label[num:, 0] = 1
    
    return blend, label

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
# Build the Discriminator
#------------------------------------------------------------------------------
def Discriminator(inputs):
    x = Conv2D(32, (3,3), padding = 'valid')(inputs)
    xx = Conv2D(64, (3,3), padding = 'valid')(x)
    x = MaxPooling2D((2,2))(xx)
    x = Flatten()(x)
    x = Dense(16, activation = 'relu')(x)  # use global average pooling to reduce size?
    x = Dense(2, activation = 'softmax')(x) # the loss must match up with this.

    return x, xx

target_input = Input(shape = input_shape_D)
#DCN_input = concatenate([SDN_out, target_input])
DCN_out = Discriminator(target_input)
DCN_label = DCN_out[0]
DCN_loss = DCN_out[1]
DCN = Model(target_input, DCN_label)
loss_net = Model(target_input, DCN_loss)

print("Summary of Discriminator:")
DCN.summary()

#------------------------------------------------------------------------------
# Pretrain each
#------------------------------------------------------------------------------
from BrainSDN import customLoss
sdn.compile(loss = customLoss, 
            optimizer = Adam(),
              )

print('-'*40)
print("Pretrain the SDN:")
print('-'*40)

sdn.fit(x_train, y_train, 
        epochs=2, batch_size= 32,
        verbose = 1,
        shuffle = True)

vis(sdn)
plt.suptitle('SDN sample after pretrained.')

X, Y = TF_blend(sdn, 380*2)
print('-'*40)
print("Pretrain the Discriminator:")
print('-'*40)
DCN.compile(loss = 'categorical_crossentropy',
            optimizer = Adam(),
            metrics=['accuracy']
            )

print(DCN.trainable)
DCN.fit(X, Y, 
        epochs = 4,
        batch_size = 32)

#------------------------------------------------------------------------------
# Build the SDN with perceptual loss
#------------------------------------------------------------------------------
set_trainability(DCN, False)
p_loss = Discriminator(SDN_out)[1]
SNDP = Model(SDN_in, p_loss)
SNDP.layers[-2].trainable= False
SNDP.layers[-1].trainable= False
SNDP.compile(loss = 'mean_squared_error',
            optimizer = Adam()
            )
print("Summary of SDNP:")
SNDP.summary()



#------------------------------------------------------------------------------
# Train the SDNP
#------------------------------------------------------------------------------

SNDP.fit(x_train, loss_net.predict(y_train),
                 epochs = 4, batch_size = 32)

vis(sdn)
vis_target()







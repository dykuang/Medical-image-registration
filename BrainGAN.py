# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 21:18:28 2018

@author: dykuang

Use a GAN model to help train the SDN
"""

import numpy as np
np.random.seed(1234)  # for reproducibility
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage.io import imread
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Dense, Flatten,UpSampling2D,\
                        multiply
from keras.utils import np_utils
from keras.optimizers import Adam, SGD
import keras.backend as K
from spatial_deformer_net import SpatialDeformer


#------------------------------------------------------------------------------
# Hyperparamters/Global setting
#------------------------------------------------------------------------------
res = 64
input_shape_G = (res, res, 2)
input_shape_D = (res, res, 1)

#------------------------------------------------------------------------------
# Data Preparation
#------------------------------------------------------------------------------
data_path = r'datasets/oasis/'
data_path_T = r'datasets/FromAnts'
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


# Reading warped images from ants
Ants = []
for temp in os.listdir(data_path_T):
     brain = imread(os.path.join(data_path_T, temp), as_grey = True)/255
     brain = resize(brain, (res,res), mode='reflect') #smooth before resample?
     Ants.append(brain)
     
Ants = np.array(Ants)
Ants = Ants.astype('float32')
Ants = np.expand_dims(Ants, 3)
#------------------------------------------------------------------------------
# Some util functions
#------------------------------------------------------------------------------
def set_trainability(model, flag = False): # need to call compile() after this?
    model.trainable = flag
    for layer in model.layers:
        layer.trainable = flag

# provide batch concatenating true and fake images
# May need to write a generator and use fit_generator
def TF_blend(G, batch_size):
    num = int(batch_size/2)
    choice = np.random.choice(380, num, replace = False) # repeated sampels when train?
    XT = Ants[choice,:]     
    XF = G.predict(x_train[choice,:])
    blend = np.concatenate((XT, XF),axis = 0)
    
    label = np.zeros((batch_size, 2))
    label[:num, 1] = 1
    label[num:, 0] = 1
    
    return blend, label
    

# provide batch with generated images
#def sample_GEN(G, batchsize = batch_size):
#    choice = np.random.choice(380, batchsize, replace = False)
#    samples = G.predict(x_train[choice])
#    
#    return samples

def sample_train(batchsize):
    choice = np.random.choice(380, batchsize, replace = False)
    samples = x_train[choice]
    label = np.zeros((batchsize, 2))
    
    label[:, 1] = 1
    return samples, label


sample_choice = np.random.choice(380, 9, replace = False)
def vis(G, choice = sample_choice):
    plt.figure()
    
    sample = G.predict(x_train[choice])
    for i in range(3):
        for j in range(3):
            plt.subplot(3, 3, 3*i+j+1)
            plt.imshow(sample[3*i+j,:,:,0])
            plt.axis('off')
#------------------------------------------------------------------------------
# Build the Generator
#------------------------------------------------------------------------------

def Generator(inputs):  
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

GEN_input = Input(shape = input_shape_G)
GEN_out = Generator(GEN_input)
GEN = Model(GEN_input, GEN_out)

print("Summary of Generator:")
GEN.summary()

#------------------------------------------------------------------------------
# Build the Discriminator
#------------------------------------------------------------------------------
def Discriminator(inputs):
#    inputs = Input(shape = input_shape)
    x = Conv2D(32, (3,3), padding = 'valid')(inputs)
    x = Conv2D(64, (3,3), padding = 'valid')(x)
    x = MaxPooling2D((2,2))(x)
    x = Flatten()(x)
    x = Dense(16, activation = 'relu')(x)  # use global average pooling to reduce size?
    x = Dense(2, activation = 'softmax')(x) # the loss must match up with this.

    return x

DCN_input = Input(shape = input_shape_D)
DCN_out = Discriminator(DCN_input)
DCN = Model(DCN_input, DCN_out)

print("Summary of Discriminator:")
DCN.summary()

#------------------------------------------------------------------------------
# Pretrain each
#------------------------------------------------------------------------------
#from BrainSDN import customLoss
#GEN.compile(loss = customLoss, 
#            optimizer = Adam(),
#              )
#
#print('-'*40)
#print("Pretrain the Generator:")
#print('-'*40)
#
#GEN.fit(x_train, y_train, 
#        epochs=1, batch_size= 32,
#        verbose = 1,
#        shuffle = True)

vis(GEN)
plt.suptitle('GEN sample after pretrained.')

X, Y = TF_blend(GEN, 380*2)
print('-'*40)
print("Pretrain the Discriminator:")
print('-'*40)
DCN.compile(loss = 'categorical_crossentropy',
            optimizer = Adam(),
            metrics=['accuracy']
            )
DCN.fit(X, Y, 
        epochs = 2,
        batch_size = 32)

#------------------------------------------------------------------------------
# Build the GAN (It may be good to have a node editor..)
#------------------------------------------------------------------------------
set_trainability(DCN, False)
GAN_out = DCN(GEN_out)
GAN = Model(GEN_input, GAN_out)
GAN.compile(loss = 'categorical_crossentropy',
            optimizer = Adam(),
            metrics=['accuracy']
            )
print("Summary of GAN:")
GAN.summary()

#------------------------------------------------------------------------------
# Train
#------------------------------------------------------------------------------

# add an argument for balancing G and D during Training?
def train(Gan, G, D, epochs, 
          batch_size, D_nt, G_nt,
          verbose = True, 
          show_freq = 50,
          d_loss=[],
          g_loss=[]):
    e_range = range(epochs)
    G_num = range(G_nt)
    D_num = range(D_nt)
#    iter_num = int(np.floor(380/batch_size))
#    iter_range = range(iter_num)
    for epoch in e_range:
#        for _ in iter_range:
            # train the discriminator, can train multiple times before passing to the next stage
        set_trainability(D, True)
        for _ in D_num:
            X_D, Y_D = TF_blend(G, batch_size)
            d_loss.append(D.train_on_batch(X_D, Y_D))
         
#        for _ in iter_range:
        set_trainability(D, False)
        for _ in G_num:
            # train the generator
            X_GAN, Y_GAN = sample_train(batch_size)
            g_loss.append(Gan.train_on_batch(X_GAN, Y_GAN))
        
        if verbose and (epoch + 1) % show_freq == 0:
                print("Epoch #{}: Generative Loss: {}, Discriminative Loss: {}".format(
                      epoch + 1, g_loss[-1], d_loss[-1]))
    return d_loss, g_loss    
    

dloss, gloss= train(GAN, GEN, DCN, 
                    epochs=5, 
                    batch_size = 128, 
                    D_nt=1,
                    G_nt = 4)

"""
TODO: Tune to Converge !!!
"""
def vis_loss(dloss, gloss):
    plt.figure()
    plt.plot(dloss)
    plt.legend(['Loss', 'Accuracy'])
    plt.title('Discriminator')
    plt.figure()
    plt.plot(gloss)
    plt.legend(['Loss', 'Accuracy'])
    plt.title('GAN')

vis(GEN)
plt.suptitle('GEN samples after trained.')
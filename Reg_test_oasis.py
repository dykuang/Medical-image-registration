# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 10:46:07 2018

@author: Dongyang


Test a single Regression Net (displacement) --> STN module on registrating 
slilces of brains from OASIS with different losses.
"""

import numpy as np
np.random.seed(1337)  # for reproducibility
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage.io import imread
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, UpSampling2D, BatchNormalization, Dense, Flatten
from keras.utils import np_utils
from keras.optimizers import Adam, SGD
from keras.losses import binary_crossentropy, kullback_leibler_divergence, mean_squared_error, mean_squared_logarithmic_error
from keras.initializers import RandomUniform
import keras.backend as K
from spatial_deformer_net import SpatialDeformer
from spatial_transformer_net import SpatialTransformer


#------------------------------------------------------------------------------
# Hyperparamters/Global setting
#------------------------------------------------------------------------------
epochs = 25
batch_size = 16
res = 60
input_shape = (res,res,2)
preprocess_flag = False

#------------------------------------------------------------------------------
# Data Preparation
#------------------------------------------------------------------------------
"""
need proprocess? just centering?
"""
data_path = r'datasets/oasis/'
import os 
train = []
for temp in os.listdir(data_path):
     brain = imread(os.path.join(data_path, temp))
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

print('x_train shape: {}'.format(x_train.shape))
print('y_train shape: {}'.format(y_train.shape))
print('-'*40) 

if preprocess_flag:
#     train_mean = np.mean(y_train, axis = 0)   
#     x_train = x_train - np.tile(train_mean, 2)
#     y_train = y_train - train_mean
     
     train_mean = np.reshape(x_train, (-1, 2)).mean(axis = 0) 
     x_train = x_train - train_mean
     y_train = y_train - train_mean[1]
#------------------------------------------------------------------------------
# NN to produce displacement field
#------------------------------------------------------------------------------
inputs = Input(shape = input_shape)

zz = Conv2D(16, (3,3), padding = 'same')(inputs)
#zz = MaxPooling2D((2,2))(zz)
zz = Conv2D(32, (3,3), padding = 'same')(zz)
#zz = UpSampling2D((2,2))(zz)    # keep the same resolution
zz = MaxPooling2D((2,2))(zz)
zz = Conv2D(64, (3,3), padding = 'same')(zz)
zz = UpSampling2D((2,2))(zz) 
zz = Conv2D(32, (3,3), padding = 'same')(zz)

zz = Conv2D(2, (3,3), padding = 'same',
                  kernel_initializer= 'zeros',
                  bias_initializer = 'zeros',
                  activation = 'tanh')(zz) #careful about the activation
locnet = Model(inputs, zz)


b = np.zeros((2, 3), dtype='float32')
b[0, 0] = 1
b[1, 1] = 1
W = np.zeros((50, 6), dtype='float32')
weights = [W, b.flatten()]

yy = MaxPooling2D((2,2))(inputs)
yy = Conv2D(20, (5,5))(yy)
yy = MaxPooling2D((2,2))(yy)
yy = Conv2D(20, (5,5))(yy)
yy = Flatten()(yy)
yy = Dense(50)(yy)
yy = Dense(6, weights=weights)(yy)
affine = Model(inputs, yy)

#------------------------------------------------------------------------------
# Custom Loss
#------------------------------------------------------------------------------

#this contains both X and Y sobel filters in the format (3,3,1,2)
#size is 3 x 3, it considers 1 input channel and has two output channels: X and Y
sobelFilter = K.variable([[[[1.,  1.]], [[0.,  2.]],[[-1.,  1.]]],
                      [[[2.,  0.]], [[0.,  0.]],[[-2.,  0.]]],
                      [[[1., -1.]], [[0., -2.]],[[-1., -1.]]]])

def expandedSobel(inputTensor):

    #this considers data_format = 'channels_last'
    inputChannels = K.reshape(K.ones_like(inputTensor[0,0,0,:]),(1,1,-1,1))
    #if you're using 'channels_first', use inputTensor[0,:,0,0] above

    return sobelFilter * inputChannels

def sobelNorm(y):
     filt = expandedSobel(y)
     sobel = K.depthwise_conv2d(y, filt)
     return K.mean(K.square(sobel))

def sobelLoss(yTrue,yPred): # this loss causes "check board" effect

    #get the sobel filter repeated for each input channel
    filt = expandedSobel(yTrue)

    #calculate the sobel filters for yTrue and yPred
    #this generates twice the number of input channels 
    #a X and Y channel for each input channel
    sobelTrue = K.depthwise_conv2d(yTrue,filt)
    sobelPred = K.depthwise_conv2d(yPred,filt)

    #now you just apply the mse:
    return K.mean(K.square(sobelTrue - sobelPred))

"""
* Add gradient loss in img_loss? may help emphasizing edges
* different forms of reg_loss?
* weights asigned for the two lossses?
"""
def customLoss(yTrue, yPred):
     img_loss = kullback_leibler_divergence(yTrue, yPred)
#     img_loss = K.sum(K.square(yTrue-yPred))
     reg_loss = sobelNorm(zz)
     
     return img_loss + sobelLoss(yTrue, yPred) + reg_loss
#------------------------------------------------------------------------------
# Training with SDN
#------------------------------------------------------------------------------

#x = SpatialTransformer(localization_net=affine,
#                             output_size=(120,120), 
#                             input_shape=input_shape)(inputs)
     
x = SpatialDeformer(localization_net=locnet,
                             output_size=(res,res), 
                             input_shape=input_shape)(inputs)

model = Model(inputs, x)
model.compile(loss = customLoss, 
              optimizer = Adam(),
              )

history = model.fit(x_train, y_train, 
                    epochs=epochs, batch_size=batch_size,
                    verbose = 0,
                    shuffle = True)

plt.figure()
plt.plot(history.history['loss'])

def see_warp(n):
    
    sample = x_train[n-1:n]
    deformed_sample = model.predict(sample)
    
    if preprocess_flag:
     #    sample = sample + np.tile(train_mean, 2)
     #    deformed_sample = deformed_sample +  train_mean
         sample = sample + train_mean
         deformed_sample = deformed_sample +  train_mean[1]
    
    plt.figure()
    plt.subplot(1,3,1)
    plt.imshow(sample[0,:,:,0])
    plt.title('moving')
    plt.axis('off')
    plt.subplot(1,3,2)
    plt.imshow(deformed_sample[0,:,:,0])
    plt.title('warped')
    plt.axis('off')
    plt.subplot(1,3,3)
    plt.imshow(sample[0,:,:,1])
    plt.title('fix')
    plt.axis('off')

see_warp(1)
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 21:34:13 2018

@author: dykuang

create a module of SDN to be used further in GAN
"""


import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage.io import imread
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D,  UpSampling2D, multiply
from keras.optimizers import Adam
from keras.losses import binary_crossentropy, kullback_leibler_divergence

import keras.backend as K
from spatial_deformer_net import SpatialDeformer


#------------------------------------------------------------------------------
# Some utility functions
#------------------------------------------------------------------------------
#this contains both X and Y sobel filters in the format (3,3,1,2)
#size is 3 x 3, it considers 1 input channel and has two output channels: X and Y
sobelFilter = K.constant([[[[1.,  1.]], [[0.,  2.]],[[-1.,  1.]]],
                      [[[2.,  0.]], [[0.,  0.]],[[-2.,  0.]]],
                      [[[1., -1.]], [[0., -2.]],[[-1., -1.]]]])

def expandedSobel(inputTensor):

    #this considers data_format = 'channels_last'
    inputChannels = K.reshape(K.ones_like(inputTensor[0,0,0,:]),(1,1,-1,1))
    #if you're using 'channels_first', use inputTensor[0,:,0,0] above

    return sobelFilter * inputChannels 

def see_warp(n):
    
    sample = x_train[n:n+1]
    deformed_sample = model.predict(sample)
    deformation = model.layers[1].locnet.predict(sample)
    
    plt.figure()
    plt.subplot(1,3,1)
    plt.imshow(sample[0,:,:,0], cmap='gray')
    plt.title('moving')
    plt.axis('off')
    plt.subplot(1,3,2)
    plt.imshow(deformed_sample[0,:,:,0],cmap='gray')
    plt.title('warped')
    plt.axis('off')
    plt.subplot(1,3,3)
    plt.imshow(sample[0,:,:,1],cmap='gray')
    plt.title('fix')
    plt.axis('off')
    
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(deformation[0,:,:,0],cmap='gray')
    plt.title('X')
    plt.axis('off')
    plt.subplot(1,2,2)
    plt.imshow(deformation[0,:,:,1],cmap='gray')
    plt.title('Y')
    plt.axis('off')
     
#------------------------------------------------------------------------------
# NN to produce displacement field
#------------------------------------------------------------------------------
def SDN(input_shape):
    inputs = Input(shape = input_shape)
    
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
                             output_size=(input_shape[0],input_shape[1]), 
                             input_shape=input_shape)(inputs)


    sdn = Model(inputs, x1)
    
    return sdn


#------------------------------------------------------------------------------
# Custom Loss
#------------------------------------------------------------------------------

def sobelNorm(y):
     filt = expandedSobel(y)
     sobel = K.depthwise_conv2d(y, filt, padding = 'same')
     
     return K.mean(K.square(sobel))

def sobelLoss(yTrue,yPred): #Consider smooth in front

    #get the sobel filter repeated for each input channel
    filt = expandedSobel(yTrue)

    #calculate the sobel filters for yTrue and yPred
    #this generates twice the number of input channels 
    #a X and Y channel for each input channel
    sobelTrue = K.depthwise_conv2d(yTrue,filt, padding = 'same')
    sobelPred = K.depthwise_conv2d(yPred,filt, padding = 'same')

    #now you just apply the mse:
    return K.mean(K.square(sobelTrue - sobelPred)), sobelTrue
#    return K.meansobelTrue*sobelPred/K.

def total_variation(y):
    assert K.ndim(y) == 4
    a = K.square(y[:, :res - 1, :res - 1, :] - y[:, 1:, :res - 1, :])
    b = K.square(y[:, :res - 1, :res - 1, :] - y[:, :res - 1, 1:, :])
    return K.mean(K.pow(a + b, 2))

def total_variation_loss(yTrue, yPred):
    assert K.ndim(yTrue) == 4
    diff = yTrue - yPred
    a = K.square(diff[:, :res - 1, :res - 1, :] - diff[:, 1:, :res - 1, :])
    b = K.square(diff[:, :res - 1, :res - 1, :] - diff[:, :res - 1, 1:, :])
    return K.mean(K.pow(a + b, 2)) # tweak the power?

"""
* Add gradient loss in img_loss? may help emphasizing edges
* different forms of reg_loss?
* weights asigned for the two lossses?
"""
def customLoss(yTrue, yPred):
     img_loss = kullback_leibler_divergence(K.reshape(yTrue, [-1])/K.sum(yTrue), K.reshape(yPred, [-1])/K.sum(yPred))
     sobel_loss, mask = sobelLoss(yTrue, yPred)
     BCE = binary_crossentropy(yTrue, yPred)
     
     return img_loss + sobel_loss + 0.3*BCE

if __name__ == '__main__':  
    #------------------------------------------------------------------------------
    # Hyperparamters/Global setting
    #------------------------------------------------------------------------------
    epochs = 25
    batch_size = 8
    res = 200
    input_shape = (res,res,2)
    preprocess_flag = False
    
    #------------------------------------------------------------------------------
    # Data Preparation
    #------------------------------------------------------------------------------
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
    
    model = SDN(input_shape)
    
    model.compile(loss = customLoss, 
              optimizer = Adam(decay=1e-5),
              )
    
    history = model.fit(x_train[:3], y_train[:3], 
                    epochs=epochs, batch_size=batch_size,
                    verbose = 0,
                    shuffle = True)

    plt.figure()
    plt.plot(history.history['loss'])
    
    
    see_warp(0)

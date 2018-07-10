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

scharrFilter = K.constant([[[[3.,  3.]], [[0.,  10.]],[[-3.,  3.]]],
                      [[[10.,  0.]], [[0.,  0.]],[[-10.,  0.]]],
                      [[[3., -3.]], [[0., -10.]],[[-3., -3.]]]])

Filter = sobelFilter

def expandedSobel(inputTensor):

    #this considers data_format = 'channels_last'
    inputChannels = K.reshape(K.ones_like(inputTensor[0,0,0,:]),(1,1,-1,1))
    #if you're using 'channels_first', use inputTensor[0,:,0,0] above

    return Filter * inputChannels 

def vis_grid(disp, direct = 2): # xy is of shape h*w*2
     
     w, h= np.shape(disp)[0], np.shape(disp)[1]
     
     x = np.linspace(-1., 1., w)
     y = np.linspace(-1., 1., h)
     
     xx , yy = np.meshgrid(x, y)
     
     xy = np.stack([xx,yy], 2) + disp
     
     plt.figure()
     
     if direct == 0:
          for row in range(w):
               x, y = xy[row,:, 0], yy[row,:]       
               plt.plot(x,y, color = 'b')
          for col in range(h):
               x, y = xy[:, col, 0], yy[:, col]       
               plt.plot(x,y, color = 'b')
               plt.ylim(1,-1)
     
     elif direct == 1:  
          for row in range(w):
               x, y = xx[row,:], xy[row,:, 1]       
               plt.plot(x,y, color = 'b')
          for col in range(h):
               x, y = xx[:, col], xy[:, col, 1]       
               plt.plot(x,y, color = 'b')
               plt.ylim(1,-1)
     
     else:
          for row in range(w):
               x, y = xy[row,:, 0], xy[row,:, 1]       
               plt.plot(x,y, color = 'b')
          for col in range(h):
               x, y = xy[:, col, 0], xy[:, col, 1]       
               plt.plot(x,y, color = 'b')
               plt.ylim(1,-1)

from sklearn.metrics import jaccard_similarity_score   
def j_score(yTrue, yPred):
     js=[]
     for yT, yP in zip(yTrue, yPred):
          js.append(jaccard_similarity_score((yT>0).flatten(), (yP>0).flatten()))
     js = np.stack(js)
     return np.mean(js)

def see_warp(n):
    
    sample = x_train[n:n+1]
    deformed_sample = sdn.predict(sample)
    deformation = sdn.layers[-1].locnet.predict(sample)
    print(K.eval(total_variation(deformation)))
    
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
    plt.imshow(deformation[0,:,:,0], cmap = 'terrain')
    plt.title('X')
    plt.axis('off')
    plt.subplot(1,2,2)
    plt.imshow(deformation[0,:,:,1], cmap = 'terrain')
    plt.title('Y')
    plt.axis('off')
    
    vis_grid(deformation[0])
    
    print('J_score:{}'.format(jaccard_similarity_score((sample[0,:,:,1]>0).flatten(),
                                                       (deformed_sample[0,:,:,0]>0).flatten())))

     
#------------------------------------------------------------------------------
# NN to produce displacement field
#------------------------------------------------------------------------------
from keras.regularizers import l1, l2
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
                      activity_regularizer = l2(0.1),
                      activation = 'tanh')(zzzz)
    
    locnet = Model(inputs, zzzz)
     
    x1 = SpatialDeformer(localization_net=locnet,
                             output_size=(input_shape[0],input_shape[1]), 
                             input_shape=input_shape)(inputs)
    
    return x1


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
    return K.mean(K.square(sobelTrue - sobelPred))


def total_variation(y):
#    assert K.ndim(y) == 4
    a = K.square(y[:, :res - 1, :res - 1, :] - y[:, 1:, :res - 1, :])
    b = K.square(y[:, :res - 1, :res - 1, :] - y[:, :res - 1, 1:, :])
    return K.pow(K.sum(a + b), 0.5) + K.pow(K.sum(K.pow(y, 2)),0.5)# tweak the power?

def total_variation_loss(yTrue, yPred):
#    assert K.ndim(yTrue) == 4
    diff = yTrue - yPred

    return total_variation(diff) + K.mean(K.square(diff))

"""
* Add gradient loss in img_loss? may help emphasize edges
* different forms of reg_loss?
* weights asigned for the two lossses?
"""
def customLoss(yTrue, yPred):
#     norm_T = K.pow(K.sum(K.square(yTrue)), 0.5)
#     norm_P = K.pow(K.sum(K.square(yPred)), 0.5)
#     img_loss = kullback_leibler_divergence(K.reshape(yTrue, [-1])/K.sum(yTrue), 
#                                            K.reshape(yPred, [-1])/K.sum(yPred))
     img_loss = kullback_leibler_divergence(K.softmax(K.reshape(yTrue, [-1])), 
                                            K.softmax(K.reshape(yPred, [-1])))
     sobel_loss = sobelLoss(yTrue, yPred)
     BCE = binary_crossentropy(yTrue, yPred)
#     return img_loss
     return img_loss + sobel_loss + 0.3*BCE

if __name__ == '__main__':  
    #------------------------------------------------------------------------------
    # Hyperparamters/Global setting
    #------------------------------------------------------------------------------
    epochs = 25
    batch_size = 8
    res = 128
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
    
    inputs = Input(shape = input_shape)
      
    sdn = Model(inputs, SDN(inputs))
    
    sdn.compile(loss = 'mse',
#                loss_weights = [1.0, 0.0],
                optimizer = Adam(decay=1e-5),
                )
#    
# =============================================================================
#    cat1 = imread('circ1.png', as_grey = True)
##    from skimage import transform as tsf
##    tform = tsf.SimilarityTransform(scale=0.8, rotation=0, translation=(0, 0))
##    cat2 = tsf.warp(cat1, tform)
#    cat2 = imread('circ.png', as_grey = True)
#    cat1 = resize(cat1, (res,res), mode='reflect')
#    cat2 = resize(cat2, (res,res), mode='reflect')
#    x_train[0,:,:,0] = cat1
#    x_train[0,:,:,1] = cat2
#      
#    y_train[0,:,:,0] = cat2
    
    history=sdn.fit(x_train[:1], y_train[:1],
            epochs = epochs, batch_size = batch_size,
            verbose = 0, shuffle = True)
   
# =============================================================================
#    from sklearn.model_selection import train_test_split
#    X_train, X_test, Y_train, Y_test = train_test_split(
#                                            x_train, y_train, test_size=0.25)
#    
#    history = sdn.fit(X_train, Y_train, 
#                    epochs=epochs, batch_size=batch_size,
#                    verbose = 0,
#                    shuffle = True)
    

#    plt.figure()
#    plt.plot(history.history['loss'])
#    
#    print('Mean J-score on test_set is {}'.format(j_score(Y_test, sdn.predict(X_test))))
    see_warp(0)

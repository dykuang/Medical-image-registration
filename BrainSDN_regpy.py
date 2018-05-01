# -*- coding: utf-8 -*-
"""
Created on Sat Apr  7 19:03:23 2018

@author: dykua

A copy of BrainSDN for testing the proper regularization loss
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

Filter = scharrFilter

def expandedSobel(inputTensor):

    #this considers data_format = 'channels_last'
    inputChannels = K.reshape(K.ones_like(inputTensor[0,0,0,:]),(1,1,-1,1))
    #if you're using 'channels_first', use inputTensor[0,:,0,0] above

    return Filter * inputChannels 

def get_jacobian(disp):
#    assert K.ndim(y) == 4

    xx = np.square(disp[:, :res - 1, :res - 1, 0] - disp[:, 1:, :res - 1, 0])
    xy = np.square(disp[:, :res - 1, :res - 1, 0] - disp[:, :res - 1, 1:, 0])
    
    yx = np.square(disp[:, :res - 1, :res - 1, 1] - disp[:, 1:, :res - 1, 1])
    yy = np.square(disp[:, :res - 1, :res - 1, 1] - disp[:, :res - 1, 1:, 1])
    
    J = xx*yy - xy*yx

    return J

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
#               plt.ylim(1,-1)
          for col in range(h):
               x, y = xy[:, col, 0], yy[:, col]       
               plt.plot(x,y, color = 'b') 
               plt.ylim(1,-1)
               plt.axis('equal')
     
     elif direct == 1:  
          for row in range(w):
               x, y = xx[row,:], xy[row,:, 1]       
               plt.plot(x,y, color = 'b')
#               plt.ylim(1,-1)
          for col in range(h):
               x, y = xx[:, col], xy[:, col, 1]       
               plt.plot(x,y, color = 'b') 
               plt.ylim(1,-1)
               plt.axis('equal')
     else:
          for row in range(w):
               x, y = xy[row,:, 0], xy[row,:, 1]       
               plt.plot(x,y, color = 'b')
          for col in range(h):
               x, y = xy[:, col, 0], xy[:, col, 1]       
               plt.plot(x,y, color = 'b') 
               plt.ylim(1,-1)
               plt.axis('equal')
               
from scipy import interpolate
# output grid displacement at different resolutions by interpolation
def grid_interpolation(grid_value, resolution = [16,16]):
     w, h= np.shape(grid_value)[1], np.shape(grid_value)[2]
     
     x = np.linspace(-1., 1., w)
     y = np.linspace(-1., 1., h)
     
     xx , yy = np.meshgrid(x, y)

     x_f = interpolate.interp2d(xx, yy, grid_value[0,:,:,0], kind='linear') 
     y_f = interpolate.interp2d(xx, yy, grid_value[0,:,:,1], kind='linear')
     
     
     x = np.linspace(-1., 1., resolution[0])
     y = np.linspace(-1., 1., resolution[1])
     
     x_val = x_f(x,y)
     y_val = y_f(x,y)
     
     return np.stack([x_val, y_val], axis = 2)     
     

from sklearn.metrics import jaccard_similarity_score   
def j_score(yTrue, yPred):
     js=[]
     for yT, yP in zip(yTrue, yPred):
          js.append(jaccard_similarity_score((yT>0.1).flatten(), (yP>0.1).flatten()))
     js = np.stack(js)
     return np.mean(js)

     
def see_warp(n):
    
    sample = x_train[n:n+1]
    deformed_sample, deformation = sdn.predict(sample)
#    print(K.eval(total_variation(deformation)))
    
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
    
    print('J-score before:\t {}'.format(j_score(sample[0,:,:,1], sample[0,:,:,0])))
    print('J-score after:\t {}'.format(j_score(sample[0,:,:,1], deformed_sample[0,:,:,0])))
    
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(deformation[0,:,:,0], cmap = 'terrain')
    plt.title('X')
    plt.axis('off')
    plt.subplot(1,2,2)
    plt.imshow(deformation[0,:,:,1], cmap = 'terrain')
    plt.title('Y')
    plt.axis('off')

    plt.figure()
    plt.imshow(deformation[0,:,:,0]**2 + deformation[0,:,:,1]**2, cmap = 'terrain')
    plt.title('Attention map')
    plt.axis('off')       
    
    vis_grid(deformation[0])


     
#------------------------------------------------------------------------------
# NN to produce displacement field
#------------------------------------------------------------------------------
from keras.regularizers import l1
from keras.layers import Conv2DTranspose
def SDN(inputs):
    
    zz = Conv2D(64, (3,3), padding = 'same')(inputs) #different stride than 1?
    zzz = Conv2D(64, (3,3), padding = 'same')(zz)
    
    zzz = MaxPooling2D((2,2))(zzz)
    zzz = Conv2D(128, (3,3), padding = 'same')(zzz)
    
    zzz = UpSampling2D((2,2))(zzz) 
    zzz = Conv2D(64, (3,3), padding = 'same')(zzz)
    
#    zzz = Conv2DTranspose(64, (2,2), strides=(2,2), padding = 'same')(zzz) # A different upsampling
    
    zzzz = multiply([zz, zzz]) 
    zzzz = Conv2D(2, (3,3), padding = 'same',
#                      kernel_initializer= 'zeros',
#                      bias_initializer = 'zeros',
#                      activity_regularizer = l1(0.001),
                      activation = 'tanh')(zzzz)
    
    locnet = Model(inputs, zzzz)
     
    x1 = SpatialDeformer(localization_net=locnet,
                             output_size=(input_shape[0],input_shape[1]), 
                             input_shape=input_shape)(inputs)
    
    return x1, locnet(inputs)

def SDN_nopooling(inputs):
    
    zz = Conv2D(64, (3,3), padding = 'same')(inputs) #different stride than 1?
    zzz = Conv2D(64, (3,3), padding = 'same')(zz)
    
    zzz = MaxPooling2D((2,2))(zzz)
    zzz = Conv2D(128, (3,3), padding = 'same')(zzz)
    
#    zzz = UpSampling2D((2,2))(zzz) 
#    zzz = Conv2D(64, (3,3), padding = 'same')(zzz)
    
    zzz = Conv2DTranspose(64, (3,3), strides=(2,2), padding = 'same')(zzz) # A different upsampling
    
    zzzz = multiply([zz, zzz]) 
    zzzz = Conv2D(2, (3,3), padding = 'same',
#                      kernel_initializer= 'zeros',
#                      bias_initializer = 'zeros',
#                      activity_regularizer = l1(0.001),
                      activation = 'tanh')(zzzz)
    
    locnet = Model(inputs, zzzz)
     
    x1 = SpatialDeformer(localization_net=locnet,
                             output_size=(input_shape[0],input_shape[1]), 
                             input_shape=input_shape)(inputs)
    
    return x1, locnet(inputs)


def SDN_deeper(inputs):
    z1_1 = Conv2D(32, (2,2), padding = 'same')(inputs)
    z1_2 = Conv2D(32, (2,2), padding = 'same')(z1_1)
    
    z2 = MaxPooling2D((2,2))(z1_2)
    z2_1 = Conv2D(64, (2,2), padding = 'same')(z2)
    z2_2 = Conv2D(64, (2,2), padding = 'same')(z2_1)
    
    z3 = MaxPooling2D((2,2))(z2_2)
    z3 = Conv2D(128, (2,2), padding = 'same')(z3)

    
    z3 = UpSampling2D((2,2))(z3) 
    z3 = Conv2D(64, (2,2), padding = 'same')(z3)   
    z4 = multiply([z2_1, z3]) 
    
    z4 = UpSampling2D((2,2))(z3)
    z4 = Conv2D(32, (2,2), padding = 'same')(z4)
    z5= multiply([z1_1, z4])    
    
 
    zzzz = Conv2D(2, (2,2), padding = 'same',
#                      kernel_initializer= 'zeros',
#                      bias_initializer = 'zeros',
#                      activity_regularizer = l1(0.001),
                      activation = 'tanh')(z5)
    
    locnet = Model(inputs, zzzz)
     
    x1 = SpatialDeformer(localization_net=locnet,
                             output_size=(input_shape[0],input_shape[1]), 
                             input_shape=input_shape)(inputs)
    
    return x1, locnet(inputs)

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
#    return K.mean(K.square(sobelTrue - sobelPred))
    return K.pow(K.sum(K.square(sobelTrue - sobelPred)), 0.5) + K.pow(K.sum(K.square(yTrue - yPred)),0.5)



def total_variation(y):
#    assert K.ndim(y) == 4
    a = K.square(y[:, :res - 1, :res - 1, :] - y[:, 1:, :res - 1, :])
    b = K.square(y[:, :res - 1, :res - 1, :] - y[:, :res - 1, 1:, :])
    
    return K.pow(K.sum(a + b), 0.5)# tweak the power?

def total_variation_loss(yTrue, yPred):
#    assert K.ndim(yTrue) == 4
    diff = yTrue - yPred

    return 10*total_variation(diff) + 0.1*K.pow(K.sum(K.pow(diff, 2)),0.5)

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
#     sobel_loss = sobelLoss(yTrue, yPred)
     BCE = binary_crossentropy(yTrue, yPred)
#     return img_loss
     return img_loss + 0.3*BCE

if __name__ == '__main__':  
    #------------------------------------------------------------------------------
    # Hyperparamters/Global setting
    #------------------------------------------------------------------------------
    epochs = 50
    batch_size = 16
    res = 100
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
    
    sdn.compile(loss = ['mse', total_variation_loss],
                loss_weights = [1.0, 0.0001],
                optimizer = Adam(decay=1e-5),
                )
#    
# =============================================================================
#    cat1 = imread('S08_095.png', as_grey = True)
##    from skimage import transform as tsf
##    tform = tsf.SimilarityTransform(scale=1.0, rotation=0, translation=(0, -10))
##    cat2 = tsf.warp(cat1, tform)
#    cat2 = imread('S09_085.png', as_grey = True)
#    cat1 = resize(cat1, (res,res), mode='reflect')
#    cat2 = resize(cat2, (res,res), mode='reflect')
#    x_train[0,:,:,0] = cat1
#    x_train[0,:,:,1] = cat2
#      
#    y_train[0,:,:,0] = cat2
##    
#    history = sdn.fit(x_train[:1], [y_train[:1], np.zeros([1,res,res,2])],
#            epochs = epochs, batch_size = batch_size,
#            verbose = 0, shuffle = True)
# =============================================================================
    from sklearn.model_selection import train_test_split
    X_train, X_test, Y_train, Y_test = train_test_split(
                                            x_train, y_train, test_size=0.25)
    
    history = sdn.fit(X_train, [Y_train, np.zeros([len(Y_train), res, res, 2])], 
                    epochs=epochs, batch_size=batch_size,
                    verbose = 0,
                    shuffle = True)
#    
#
    plt.figure()
    plt.plot(history.history['loss'])
#    
    print('Mean J-score on test_set is {}'.format(j_score(Y_test, sdn.predict(X_test)[0])))
    see_warp(0)
    
#    plt.figure()
#    disp = sdn.layers[-1].predict(x_train[:1])
#    J = get_jacobian(disp)
#    plt.imshow(J[0])


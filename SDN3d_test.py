# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 16:34:21 2018

@author: Dongyang

This script tests spatial_deformer_net3d.py
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

'''
TODO:
     Tune it.  Training does not converge....
'''

# Create/Load data
x, y, z = np.indices([8,8,8])
#xx, yy, zz = np.indices([4,4,4])
#voxels = np.zeros_like(x)
#voxels[4] = 0

vol1 = ((x>1) & (y>1) & (z>1)) & ((x<5) & (y<5) & (z<5)).astype(int) # weird, block between 1 & 3 actually
#vol1 = voxels[:4,:4,:4]
#vol1[1:2,1:2,1:2] = 1

vol2 = ((x>2) & (y>1) & (z>1)) & ((x<6) & (y<5) & (z<5)).astype(int)
#vol2 = voxels[:4,:4,:4]
#vol2[2:3, 2:3, 2:3] = 1

#vol1 = voxels[:4,:4,:4]
#vol2 = voxels[1:,1:,1:]

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.voxels(vol1,facecolors='b', edgecolor='k')

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.voxels(vol2,facecolors='b', edgecolor='k')

# Training
from spatial_deformer_net3d import SpatialDeformer3D
from keras.models import Model
from keras.layers import Input, Conv3D, MaxPooling3D,  UpSampling3D, multiply
from keras.optimizers import Adam
from keras.losses import binary_crossentropy, kullback_leibler_divergence
import keras.backend as K
from keras.regularizers import l1, l2

input_shape = (8,8,8,2)
def SDN(inputs):
    
    zz = Conv3D(64, (2,2,2), padding = 'same')(inputs)
    zzz = Conv3D(64, (2,2,2), padding = 'same')(zz)
    
    zzz = MaxPooling3D((2,2,2))(zzz)
    zzz = Conv3D(128, (2,2,2), padding = 'same')(zzz)
    
    zzz = UpSampling3D((2,2,2))(zzz) 
    zzz = Conv3D(64, (2,2,2), padding = 'same')(zzz)
    
    zzzz = multiply([zz, zzz])   # easy to cause trouble when shape does not contain enough power of 2.
    zzzz = Conv3D(3, (2,2,2), padding = 'same',
                      kernel_initializer= 'zeros',
                      bias_initializer = 'zeros',
#                      activity_regularizer = l2(0.1),
                      activation = 'tanh')(zzz)
    
    locnet = Model(inputs, zzzz)
     
    x1 = SpatialDeformer3D(localization_net=locnet,
                             output_size=(input_shape[0],input_shape[1], input_shape[2]), 
                             input_shape=input_shape)(inputs)
    
    return x1

inputs = Input(shape = input_shape)
      
sdn = Model(inputs, SDN(inputs))
    
sdn.compile(loss = 'mse',optimizer = Adam(decay=1e-5) )

x_train = np.expand_dims(np.stack([vol1, vol2], axis = 3), 0)
y_train = x_train[...,1]
y_train = np.expand_dims(y_train, 4)

epochs = 5
batch_size = 4
history = sdn.fit(x_train, y_train,
            epochs = epochs, batch_size = batch_size,
            verbose = 0, shuffle = True)

# visualize
#plt.plot(history.history['loss'])
warped = sdn.predict(x_train)[0,...,0]
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.voxels(warped,facecolors='b', edgecolor='k')
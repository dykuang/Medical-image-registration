# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 16:34:21 2018

@author: Dongyang

This script tests spatial_deformer_net3d.py
"""
import numpy as np
import random
#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D

'''
TODO:
     * Investigate a proper loss
     * Think about a nice visualization
       * mask out those voxels with low intensity? (post-process)
       * Visualize the 'difference': on warped image with difference as the color/transparancy?
     * DIfferent architecture
     * Preprocess/postprocess
       *Smooth the image before training?
'''

#num = 3
#res = 224 
par = {'res1': 91,

       'res2': 109,

       'res3': 91,

       'kernel_size': (2,2,2),

       'kernel_strides': 2,

       'loss_weights': [1, 2e-3],

       'epochs': 10,

       'batch_size': 6,

       'lr': 1e-3, 

       'w1': 3,

       'w2': 1 
       }

print(par)

res1 = par['res1']

res2 = par['res2']

res3 = par['res3']

#===============================================================================
# Write a generator, better not use double 'for' within the body
def vol_generator(path, file_list, batch_size):
    x = np.zeros((batch_size, res1, res2, res3, 2))
    zeros = np.zeros([batch_size, res1, res2, res3, 3])
    count = 0
    while True:
#        for i in range(int(len(file_list)/batch_size)):
        for j in range(batch_size):
            x[j] = np.load(path+file_list[(count*batch_size+j)%len(file_list)])
                #x[j] = x[j]*(x[j]>0.1)
        yield x, [np.expand_dims(x[...,1],4), zeros]
       
        count = count + 1
        if count >= len(file_list)/batch_size:
            count = 0
        # random.shuffle(file_list)
        #print('list_shuffled.')

def vol_generator2(path, file_list, batch_size):
    x = np.zeros((batch_size, res1, res2, res3, 2))
    zeros = np.zeros([batch_size, res1, res2, res3, 3])
    count = 0
    while True:
        for j in range(batch_size):
            pair = file_list[(count*batch_size+j)%len(file_list)]
            mov = np.load(path+pair[:2]+'.npy')
            fix = np.load(path+pair[2:]+'.npy')
            x[j] = np.stack([mov, fix], axis = 3)

        yield x, [np.expand_dims(x[...,1],4), zeros]

        count = count + 1
        if count >= len(file_list)/batch_size:
            count = 0

#def get_batch(datapath, file_list, batchnum, batch_size):
#    x = np.zeros((batch_size, res1, res2, res3, 2))
#    zeros = np.zeros([batch_size, res1, res2, res3, 3])
#    for j in range(batch_size):
#        temp = np.load(datapath+file_list[(batchnum*batch_size+j)%len(file_list)])
        #print(temp.shape)
#        x[j] = temp
        #x[j] = x[j]*(x[j]>0.05)
#    return  x, [np.expand_dims(x[...,1],4), zeros]


# Training
from spatial_deformer_net3d import SpatialDeformer3D
from keras.models import Model
from keras.layers import Input, Conv3D, MaxPooling3D,  UpSampling3D, multiply, Conv3DTranspose, ZeroPadding3D, Cropping3D, BatchNormalization
from keras.optimizers import Adam
from keras.losses import binary_crossentropy, kullback_leibler_divergence
import keras.backend as K
from keras.regularizers import l1, l2


#from Utils import vis_grid_3d
    
def total_variation(y):
#    assert K.ndim(y) == 4
    a = K.square(y[:, :res1 - 1, :res2 - 1, :res3-1, :] - y[:, 1:, :res2 - 1, :res3-1, :])
    b = K.square(y[:, :res1 - 1, :res2 - 1, :res3-1, :] - y[:, :res1 - 1, 1:, :res3-1, :])
    c = K.square(y[:, :res1 - 1, :res2 - 1, :res3-1, :] - y[:, :res1 - 1, :res2-1, 1:, :])
    
    return K.pow(K.sum(a + b + c), 0.5)# tweak the power?

def total_variation_loss(yTrue, yPred):
#    assert K.ndim(yTrue) == 4
    diff = yTrue - yPred

    return par['w1']*total_variation(diff) + par['w2']*K.pow(K.sum(K.pow(diff, 2)),0.5)


def customloss(yTrue, yPred):
     sse = K.sum(K.square(yTrue - yPred))
     
     Dx_yTrue = yTrue[:, :res1 - 1, :res2 - 1, :res3-1, :] - yTrue[:, 1:, :res2 - 1, :res3-1, :]
     Dy_yTrue = yTrue[:, :res1 - 1, :res2 - 1, :res3-1, :] - yTrue[:, :res1 - 1, 1:, :res3-1, :]
     Dz_yTrue = yTrue[:, :res1 - 1, :res2 - 1, :res3-1, :] - yTrue[:, :res1 - 1, :res2-1, 1:, :]
     
     Dx_yPred = yPred[:, :res1 - 1, :res2 - 1, :res3-1, :] - yPred[:, 1:, :res2 - 1, :res3-1, :]
     Dy_yPred = yPred[:, :res1 - 1, :res2 - 1, :res3-1, :] - yPred[:, :res1 - 1, 1:, :res3-1, :]
     Dz_yPred = yPred[:, :res1 - 1, :res2 - 1, :res3-1, :] - yPred[:, :res1 - 1, :res2-1, 1:, :]
     
     D1 = K.sum(K.square(Dx_yTrue - Dx_yPred))
     D2 = K.sum(K.square(Dy_yTrue - Dy_yPred))
     D3 = K.sum(K.square(Dz_yTrue - Dz_yPred))
     
     return sse+ 0.2*(D1+D2+D3)

input_shape = (res1,res2,res3,2)

import tensorflow as tf
def cc3D(win=[9, 9, 9], voxel_weights=None): # a way to pass additional argument
    def loss(I, J):
        I2 = I*I
        J2 = J*J
        IJ = I*J

        filt = tf.ones([win[0], win[1], win[2], 1, 1]) 

        I_sum = tf.nn.conv3d(I, filt, [1, 1, 1, 1, 1], "SAME")
        J_sum = tf.nn.conv3d(J, filt, [1, 1, 1, 1, 1], "SAME")
        I2_sum = tf.nn.conv3d(I2, filt, [1, 1, 1, 1, 1], "SAME")
        J2_sum = tf.nn.conv3d(J2, filt, [1, 1, 1, 1, 1], "SAME")
        IJ_sum = tf.nn.conv3d(IJ, filt, [1, 1, 1, 1, 1], "SAME")

        win_size = win[0]*win[1]*win[2]
        u_I = I_sum/win_size
        u_J = J_sum/win_size

        cross = IJ_sum - u_J*I_sum - u_I*J_sum + u_I*u_J*win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I*u_I*win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J*u_J*win_size

        cc = cross*cross / (I_var*J_var+1e-5)

        # if(voxel_weights is not None):
        #	cc = cc * voxel_weights

        #return -tf.log(tf.reduce_mean(cc))
        return 1/tf.reduce_mean(cc)-1
    return loss

from architecture import SDN_ver12 as SDN
#from keras.layers import GaussianNoise
inputs = Input(shape = input_shape)
#aug = GaussianNoise(0.05)(inputs)
disp_M = Model(inputs, SDN(inputs))
warped = SpatialDeformer3D(localization_net=disp_M, output_size=(input_shape[0],input_shape[1], input_shape[2]), input_shape=input_shape)(inputs)

sdn = Model(inputs, [warped, disp_M(inputs)])
   
#print(sdn.summary())
print(sdn.layers[-1].summary())
#print(sdn.layers)
sdn.compile(loss = [cc3D(), total_variation_loss],
            loss_weights = par['loss_weights'],
            optimizer = Adam(lr = par['lr'], decay = 1e-5) )

epochs = par['epochs'] 
batch_size = par['batch_size'] 

datapath = r'/LPBA40_npy/image/'
labelpath = r'/LPBA40_npy/label/'
testpath = r'/LPBA40_npy/image_test/'

# if using vol_generator2
train_list = []
validation_list=[]

from itertools import combinations
for ind in combinations(range(1,31,1),2):
    train_list.append('{0:02d}{1:02d}'.format(ind[0],ind[1]))
    train_list.append('{0:02d}{1:02d}'.format(ind[1],ind[0]))
for ind in combinations(range(31,41,1),2):
    validation_list.append('{0:02d}{1:02d}'.format(ind[0],ind[1]))
    validation_list.append('{0:02d}{1:02d}'.format(ind[1],ind[0]))

#datapath = testpath
#train_list = validation_list
#train_list = ['3239']
#validation_list = ['3239']
'''
Training stucks at providing data without going to actual training. Did not know why...

'''
#train_list = train_list[:320]
#validation_list = validation_list[:25]

gen_train = vol_generator2(datapath, train_list, batch_size)
gen_test = vol_generator2(testpath, validation_list, batch_size)
history = sdn.fit_generator(gen_train, steps_per_epoch = len(train_list)/batch_size, epochs = epochs, use_multiprocessing = True, verbose=1, validation_data = gen_test, validation_steps = len(validation_list)/batch_size)
loss = history.history['loss']
val_loss = history.history['val_loss']
print("Training complete.")
print("Saving current model ...")
sdn.layers[-1].save_weights('/output/SDN3d_weights.h5') 





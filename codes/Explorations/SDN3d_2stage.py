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

       'loss_weights_1': [1, 5e-6],
      
       'loss_weights_2': [1, 2e-5],

       'epochs': 5,

       'batch_size': 16,

       'lr1': 1e-3,
       
       'lr2': 5e-4, 

       'w1': 0.5,

       'w2': 1 
       }

print(par)

res1 = par['res1']

res2 = par['res2']

res3 = par['res3']

# Create/Load data
# =============================================================================
# y, x, z = np.indices([res,res,res]) # use y-x-z coordinate
# 
# 
# vol1 = ((x>=2) & (y>=2) & (z>=2)) & ((x<5) & (y<5) & (z<5)).astype(int) 
# 
# 
# vol2 = ((x>=3) & (y>=2) & (z>=2)) & ((x<6) & (y<5) & (z<5)).astype(int)
# 
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax.voxels(vol1,facecolors='b', edgecolor='k')
# 
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax.voxels(vol2,facecolors='b', edgecolor='k')
# =============================================================================
#index = ["{0:02d}".format(i) for i in range(1,1+num,1)]
#import SimpleITK as sitk
#datapath = r'/home/dkuang/LPBA40/delineation_resize/'
# try if reading all together, memory error
#from itertools import combinations
#train = np.zeros([num*(num-1),res,res,res, 2])
#for i, ind in enumerate(combinations(range(num),2)):
#	mov = sitk.ReadImage(datapath+'S{}.nii.gz'.format(index[ind[0]]))
#	fix = sitk.ReadImage(datapath+'S{}.nii.gz'.format(index[ind[1]]))
#	train[i,...,0] = sitk.GetArrayViewFromImage(mov)
#	train[i,...,1] = sitk.GetArrayViewFromImage(fix)
#	train[i+int(num*(num-1)/2),...,0] = sitk.GetArrayViewFromImage(fix)
#	train[i+int(num*(num-1)/2),...,1] = sitk.GetArrayViewFromImage(mov)
#y_train = np.expand_dims(train[...,1],4)
#print(y_train.shape)

#from sklearn.model_selection import train_test_split

#X_train_ind, X_test_ind, Y_train_ind, Y_test_ind = train_test_split(                                            			range(int(num*(num-1))), range(int(num*(num-1))), test_size=0.5)

#X_train = train[X_train_ind]
#X_test = train[X_test_ind]
#Y_train = y_train[Y_train_ind]
#Y_test = y_train[Y_test_ind]
#print(X_train.shape)
#print(Y_train.shape)
#===============================================================================
# Write a generator, better not use double 'for' within
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


#==============================================================================
#from skimage.transform import resize
#brain1_path = r'/home/dkuang/LPBA40/delineation_space/S{}/S{}.delineation.structure.label.hdr'.format(index[0], index[0])
#brain2_path = r'/home/dkuang/LPBA40/delineation_space/S{}/S{}.delineation.structure.label.hdr'.format(index[1], index[1])
#brain1_path = r'C:\Users\Dongyang\Downloads\LPBA40\delineation_space\S09\S09.delineation.skullstripped.hdr'
#brain2_path = r'C:\Users\Dongyang\Downloads\LPBA40\delineation_space\S08\S08.delineation.skullstripped.hdr'
#brain1 = sitk.ReadImage(brain1_path)
#brain1_data = sitk.GetArrayViewFromImage(brain1)
#brain2 = sitk.ReadImage(brain2_path)
#brain2_data = sitk.GetArrayViewFromImage(brain2)
#brain1_data = resize(brain1_data/np.max(brain1_data), [res, res, res]) # could cause problems
#brain2_data = resize(brain2_data/np.max(brain2_data), [res, res, res])

#fig = plt.figure()
#ax = fig.gca(projection='3d')
#color = np.stack((brain1_data,brain1_data,brain1_data, brain1_data**0.5),axis=3)
#ax.voxels(brain1_data,facecolors=color, edgecolor='k', linestyle = '--',
#          linewidth=0.1)

#fig = plt.figure()
#ax = fig.gca(projection='3d')
#color = np.stack((brain2_data,brain2_data,brain2_data,brain2_data**0.5),axis=3)
#ax.voxels(brain2_data,facecolors=color, edgecolor='k', linestyle = '--',
#          linewidth=0.1)

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
from keras.layers import PReLU, LeakyReLU,add
def SDN_ver1(inputs): #should control the size carefully, larger strides to downsample 
    #z1_1 = Conv3D(32, (2,2,2), padding = 'same')(inputs)
    z1_2 = Conv3D(32, (2,2,2), strides = 2, padding = 'valid')(inputs)
    z1_2 = PReLU(shared_axes = [4])(z1_2)    
    #z2_1 = Conv3D(64, (2,2,2), padding = 'same')(z1_2)
    z2_2 = Conv3D(64, (2,2,2), strides = 2, padding = 'valid')(z1_2)
    z2_2 = PReLU(shared_axes = [4])(z2_2)
    #z3 = Conv3D(64, (2,2,2), padding = 'same')(z2_2)
    #z3 = PReLU()(z3)
    z4 = Conv3DTranspose(64, (2,2,2), strides=2, padding = 'valid')(z2_2)
    #z4 = PReLU(shared_axes = [4])(z4)
    z4 = Conv3D(64, (2,2,2), padding = 'same')(z4)
   #z4 = PReLU()(z4)
    z5 = Conv3DTranspose(32, (2,2,2), strides=2, padding = 'valid')(z4)   
    z5 = Conv3D(32, (2,2,2), padding = 'same', activation = 'linear')(z5)
    #z5 = PReLU()(z5)  
    z5 = ZeroPadding3D((2,1,2))(z5)     #Extra padding to make size match
    zzzz = Conv3D(3, (2,2,2), padding = 'valid',
#                      kernel_initializer= 'he_uniform',
#                      bias_initializer = 'he_uniform',
#                      activity_regularizer = l1(0.001),
                      activation = 'tanh')(z5)
    
    locnet = Model(inputs, zzzz)
     
    x1 = SpatialDeformer3D(localization_net=locnet,
                             output_size=(input_shape[0],input_shape[1], input_shape[2]),
                             input_shape=input_shape)(inputs)
    
    return x1, locnet(inputs)

from keras.layers import concatenate

def SDN_ver2(inputs): #should control the size carefully, larger strides to downsample 

#    z1_1 = Conv3D(32, par['kernel_size'], padding = 'same')(inputs)

    z1_2 = Conv3D(32, par['kernel_size'], strides = par['kernel_strides'], padding = 'valid', activation = 'linear')(inputs)
    #z1_2 = BatchNormalization()(z1_2)
    z1_2 = PReLU(shared_axes = [4])(z1_2)
#    z2_1 = Conv3D(64, par['kernel_size'], padding = 'same')(z1_2)

    z2_2 = Conv3D(64, par['kernel_size'], strides = par['kernel_strides'], padding = 'valid', activation = 'linear')(z1_2)
   # z2_2 = BatchNormalization()(z2_2)  
    z2_2 = PReLU(shared_axes = [4])(z2_2)
    
    z4x = Conv3DTranspose(64, par['kernel_size'], strides= par['kernel_strides'], padding = 'valid', activation = 'linear')(z2_2)
    z4x = Conv3D(64, (2,2,2), padding = 'same')(z4x)
    z5x = Conv3DTranspose(32, par['kernel_size'], strides= par['kernel_strides'], padding = 'valid', activation = 'linear')(z4x)   
    z5x = Conv3D(32, (2,2,2), padding = 'same')(z5x)
    z5x = ZeroPadding3D((2,1,2))(z5x)
    zzzzx = Conv3D(1, par['kernel_size'], padding = 'valid', activation = 'tanh')(z5x)

    z4y = Conv3DTranspose(64, par['kernel_size'], strides= par['kernel_strides'], padding = 'valid',activation = 'linear')(z2_2)
    z4y = Conv3D(64, (2,2,2), padding = 'same')(z4y)

    z5y = Conv3DTranspose(32, par['kernel_size'], strides= par['kernel_strides'], padding = 'valid', activation = 'linear')(z4y)   
    z5y = Conv3D(32, (2,2,2), padding = 'same')(z5y)

    z5y = ZeroPadding3D((2,1,2))(z5y)
    zzzzy = Conv3D(1, par['kernel_size'], padding = 'valid', activation = 'tanh')(z5y)

    z4z = Conv3DTranspose(64, par['kernel_size'], strides= par['kernel_strides'], padding = 'valid', activation = 'linear')(z2_2)
    z4z = Conv3D(64, (2,2,2), padding = 'same')(z4z)

    z5z = Conv3DTranspose(32, par['kernel_size'], strides= par['kernel_strides'], padding = 'valid', activation = 'linear')(z4z)   
    z5z = Conv3D(32, (2,2,2), padding = 'same')(z5z)
    z5z = ZeroPadding3D((2,1,2))(z5z)
    zzzzz = Conv3D(1, par['kernel_size'], padding = 'valid', activation = 'tanh')(z5z)   

    zzzz = concatenate([zzzzx, zzzzy, zzzzz], axis = -1)   

    locnet = Model(inputs, zzzz)    

    x1 = SpatialDeformer3D(localization_net=locnet, output_size=(input_shape[0],input_shape[1], input_shape[2]), input_shape=input_shape)(inputs)   

    return x1, locnet(inputs)

def SDN_ver3(inputs): #should control the size carefully, larger strides to downsample 

#    z1_1 = Conv3D(32, par['kernel_size'], padding = 'same')(inputs)

    z1_2 = Conv3D(32, par['kernel_size'], strides = par['kernel_strides'], padding = 'valid', activation = 'linear')(inputs)
    #z1_2 = BatchNormalization()(z1_2)
    z1_2 = PReLU(shared_axes = [4])(z1_2)
#    z2_1 = Conv3D(64, par['kernel_size'], padding = 'same')(z1_2)

    z2_2 = Conv3D(64, par['kernel_size'], strides = par['kernel_strides'], padding = 'valid', activation = 'linear')(z1_2)
   # z2_2 = BatchNormalization()(z2_2)  
    z2_2 = PReLU(shared_axes = [4])(z2_2)

    z4 = Conv3DTranspose(64, par['kernel_size'], strides= par['kernel_strides'], padding = 'valid', activation = 'linear')(z2_2)
    z4 = Conv3D(64, (2,2,2), padding = 'same')(z4)

    z5x = Conv3DTranspose(32, par['kernel_size'], strides= par['kernel_strides'], padding = 'valid', activation = 'linear')(z4)
    z5x = Conv3D(32, (2,2,2), padding = 'same')(z5x)
    z5x = ZeroPadding3D((2,1,2))(z5x)
    zzzzx = Conv3D(1, par['kernel_size'], padding = 'valid', activation = 'tanh')(z5x)

    z5y = Conv3DTranspose(32, par['kernel_size'], strides= par['kernel_strides'], padding = 'valid', activation = 'linear')(z4)
    z5y = Conv3D(32, (2,2,2), padding = 'same')(z5y)
    z5y = ZeroPadding3D((2,1,2))(z5y)
    zzzzy = Conv3D(1, par['kernel_size'], padding = 'valid', activation = 'tanh')(z5y)

    z5z = Conv3DTranspose(32, par['kernel_size'], strides= par['kernel_strides'], padding = 'valid', activation = 'linear')(z4)
    z5z = Conv3D(32, (2,2,2), padding = 'same')(z5z)
    z5z = ZeroPadding3D((2,1,2))(z5z)
    zzzzz = Conv3D(1, par['kernel_size'], padding = 'valid', activation = 'tanh')(z5z)

    zzzz = concatenate([zzzzx, zzzzy, zzzzz], axis = -1)

    locnet = Model(inputs, zzzz)

    x1 = SpatialDeformer3D(localization_net=locnet, output_size=(input_shape[0],input_shape[1], input_shape[2]), input_shape=input_shape)(inputs)

    return x1, locnet(inputs)

def SDN_ver4(inputs):  
    z1_1 = Conv3D(32, (2,2,2), padding = 'same')(inputs)
    z1_2 = Conv3D(32, (2,2,2), strides = 2, padding = 'valid')(z1_1)
    z1_2 = PReLU(shared_axes = [4])(z1_2)
    
    z2_1 = Conv3D(64, (2,2,2), padding = 'same')(z1_2)
    z2_2 = Conv3D(64, (2,2,2), strides = 2, padding = 'valid')(z2_1)
    z2_2 = PReLU(shared_axes = [4])(z2_2)
    
    z4 = Conv3DTranspose(64, (2,2,2), strides=2, padding = 'valid')(z2_2)
    #z4 = PReLU(shared_axes = [4])(z4)
    z4 = Conv3D(64, (2,2,2), padding = 'same')(z4)
    #z4 = PReLU()(z4)
    z5 = Conv3DTranspose(32, (2,2,2), strides=2, padding = 'valid')(z4)
    z5 = Conv3D(32, (2,2,2), padding = 'same', activation = 'linear')(z5)
    #z5 = PReLU()(z5)  
    z5 = ZeroPadding3D((2,1,2))(z5)     #Extra padding to make size match
    zzzz = Conv3D(3, (2,2,2), padding = 'valid',
#                      kernel_initializer= 'he_uniform',
#                      bias_initializer = 'he_uniform',
#                      activity_regularizer = l1(0.001),
                      activation = 'tanh')(z5)

    locnet = Model(inputs, zzzz)

    x1 = SpatialDeformer3D(localization_net=locnet,
                             output_size=(input_shape[0],input_shape[1], input_shape[2]),
                             input_shape=input_shape)(inputs)

    return x1, locnet(inputs)

def SDN_ver5(inputs):  
    #z1_1 = Conv3D(32, (2,2,2), padding = 'same')(inputs)
    z1_2 = Conv3D(32, (2,2,2), strides = 2, padding = 'valid')(inputs)
    z1_2 = PReLU(shared_axes = [4])(z1_2)

    #z2_1 = Conv3D(64, (2,2,2), padding = 'same')(z1_2)
    z2_2 = Conv3D(64, (2,2,2), strides = 2, padding = 'valid')(z1_2)
    z2_2 = PReLU(shared_axes = [4])(z2_2)

    z4 = Conv3DTranspose(32, (2,2,2), strides=2, padding = 'valid')(z2_2)
    z4 = ZeroPadding3D(((1,0), (0,0), (1,0)))(z4)
    z4 = add([z4, z1_2])
    z4 = PReLU(shared_axes = [4])(z4)
    z4 = Conv3D(32, (2,2,2), padding = 'same')(z4)
   #z4 = PReLU()(z4)

    z5 = Conv3DTranspose(64, (2,2,2), strides=2, padding = 'valid')(z4)
    z5 = Conv3D(64, (2,2,2), padding = 'same', activation = 'linear')(z5)
    #z5 = PReLU()(z5)  
    z5 = ZeroPadding3D(((1,0),(1,0),(1,0)))(z5)     #Extra padding to make size match
    zzzz = Conv3D(3, (2,2,2), padding = 'same',
#                      kernel_initializer= 'he_uniform',
#                      bias_initializer = 'he_uniform',
#                      activity_regularizer = l1(0.001),
                      activation = 'tanh')(z5)

    locnet = Model(inputs, zzzz)

    x1 = SpatialDeformer3D(localization_net=locnet,
                             output_size=(input_shape[0],input_shape[1], input_shape[2]),
                             input_shape=input_shape)(inputs)

    return x1, locnet(inputs)



inputs = Input(shape = input_shape)
      
sdn = Model(inputs, SDN_ver1(inputs))

#print(sdn.summary())
print(sdn.layers[-1].summary())
#print(sdn.layers)
from keras.optimizers import Adam
sdn.compile(loss = ['mse', total_variation_loss],
            loss_weights = par['loss_weights_1'],
            optimizer = Adam(lr = par['lr1'], decay = 1e-5) )

#x_train = np.expand_dims(np.stack([brain1_data, brain2_data], axis = 3), 0)
#y_train = x_train[...,1]
#y_train = np.expand_dims(y_train, 4)


import os
datapath = r'/LPBA40_npy_flirt/image/'
labelpath = r'/LPBA40_npy_flirt/label/'
testpath = r'/LPBA40_npy_flirt/image_test/'
datalist = os.listdir(datapath)
testlist = os.listdir(testpath)
#print(datalist)
#random.shuffle(datalist)
#train_list = datalist[:]
#test_list = datalist[:1]
#train_list = ['0520.npy']
#test_list = ['0520.npy']
#print(train_list)
#print(len(train_list))
#validation_list = testlist[:]

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


'''
Training stucks at providing data without going to actual training. Did not know why...
 
'''

gen_train = vol_generator2(datapath, train_list, par['batch_size'])
gen_test = vol_generator2(testpath, validation_list, par['batch_size'])

history1 = sdn.fit_generator(gen_train, steps_per_epoch = len(train_list)/par['batch_size'], epochs = par['epochs'], use_multiprocessing = True, verbose=1, validation_data = gen_test, validation_steps = len(validation_list)/par['batch_size'])
loss = history1.history['loss']
val_loss = history1.history['val_loss']
print("Training stage 1 complete.")

sdn.compile(loss = ['mse', total_variation_loss],
            loss_weights = par['loss_weights_2'],
            optimizer = Adam(lr = par['lr2'], decay = 1e-5) )

history2 = sdn.fit_generator(gen_train, steps_per_epoch = len(train_list)/par['batch_size'], epochs = par['epochs'], use_multiprocessing = True, verbose=1, validation_data = gen_test, validation_steps = len(validation_list)/par['batch_size'])

loss = history1.history['loss'] + history2.history['loss']
val_loss = history1.history['val_loss'] + history2.history['val_loss']

print("Training stage 2 complete.")
print("Saving current model ...")
sdn.layers[-1].save_weights('/output/SDN3d_weights.h5') 

"""
Now try using train_on_batch
"""
#loss=[]
#for i in range(epochs):
#    print("{0}/{1} epoch:\n".format(i+1, epochs))
    #print(int(np.floor(len(train_list)/batch_size)))
#    for j in range(int(np.floor(len(train_list)/batch_size))):
#        X, Y = get_batch(datapath, train_list, j, batch_size)
        #print(X[0].shape)
        #print(Y[0].shape)
#        history = sdn.train_on_batch(X, Y)
#        loss.append(history)
#        print("{}-th, minibatch done.\n".format(j+1))
#    random.shuffle(train_list)

#print("Saving current model ...")
#sdn.save('SDN3d.h5')
#trainlist = open('trainlist.txt', 'w')
#for f in train_list:
    #trainlist.write(f)
    #trainlist.write("\n")
#trainlist.close()

#testlist = open('testlist.txt', 'w')
#for f in test_list:
    #testlist.write(f)
    #testlist.write("\n")
#testlist.close()

#loss_history = open('result_test/loss_history.txt', 'w')
#for item in loss:
    #loss_history.write("%s\n"% item)
#loss_history.close()


from sklearn.metrics import jaccard_similarity_score
def j_score(yTrue, yPred):
     js=[]
     for yT, yP in zip(yTrue, yPred):
          js.append(jaccard_similarity_score((yT>0.1).flatten(), (yP>0.1).flatten()))
     js = np.stack(js)
     return np.mean(js)

# test on one brain
label = '3132.npy' 

brain_test1_mov = np.load(testpath+label[:2]+'.npy')
brain_test1_fix = np.load(testpath+label[2:4]+'.npy')
brain_test1 = np.stack([brain_test1_mov, brain_test1_fix], 3)


brain_test_label1 = np.load(labelpath+label[:2]+'.npy')
brain_test_label2 = np.load(labelpath+label[2:4]+'.npy')

def Dice(y_true, y_pred):
     T = (y_true.flatten()>0)
     P = (y_pred.flatten()>0)  

     return 2*np.sum(T*P)/(np.sum(T) + np.sum(P))


label1_data = np.rint(brain_test_label1)
label2_data = np.rint(brain_test_label2)

label_list = list(np.unique(label1_data))[1:]


dice_before, dice_after = [], []

'''
Different ways to warp the label mask makes huge differences...
'''
brain_test1 = np.expand_dims(brain_test1, 0)
warped_brain, deformation = sdn.predict(brain_test1)

from Utils import transform
for i in label_list:

     dice_before.append(Dice(label2_data==i, label1_data == i))
     seg = transform(deformation, np.expand_dims(np.expand_dims(label1_data==i, 3),0), (res1, res2, res3))  

     dice_after.append(Dice(label2_data == i, np.rint(seg)>0))
     #print("ROI {} result appended..".format(i))

count_worse=0
count_equal = 0
count_better = 0

for i in range(56):

    if dice_after[i] < dice_before[i]:

        count_worse += 1

    elif dice_after[i] > dice_before[i]:

        count_better += 1

    else:

        count_equal += 1

print('worse: {}'.format(count_worse))
print('equal: {}'.format(count_equal))
print('better: {}'.format(count_better))

print("writing to log....")
test_log = open('/output/SDN3dTest_tune_sep.txt', 'w')
test_log.write("paramters:\n")
for key, value in par.items():
    test_log.write("{}:\t{}\n".format(key, value))
test_log.write("*"*40+"\n")

test_log.write("training loss:\n")
for item in loss:
    test_log.write("%s\n"% item)
test_log.write("validation loss:\n")
for item in val_loss:
    test_log.write("%s\n"% item)

test_log.write("Brain pair: {}\n".format(label))
test_log.write("Before registration: {}\n".format(j_score(brain_test1[...,1], brain_test1[...,0])))
test_log.write("After registration: {}\n".format(j_score(brain_test1[...,1], warped_brain)))
test_log.write("Before registration: {}\n".format(Dice(brain_test1[...,1], brain_test1[...,0])))
test_log.write("After registration: {}\n".format(Dice(brain_test1[...,1], warped_brain)))
test_log.write("*"*40+"\n")
test_log.write("Before Registration"+"\t" + "After Registration\n")
for before, after in zip(dice_before, dice_after):
    test_log.write("{0:.6f}".format(before)+"\t" + "{0:.6f}".format(after)+'\n')

test_log.close()



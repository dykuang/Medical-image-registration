#-*- coding: utf-8 -*-
"""
Created on Tue Apr 17 16:34:21 2018

@author: Dongyang

A copy from spatial_deformer_net3d.py for tuning
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

num = 2
res = 224 
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
# Write a generator
def vol_generator(datapath, file_list, batch_size):
    x = np.zeros((batch_size, res, res, res, 2))
    while True:
        for i in range(int(len(file_list)%batch_size)):
            for j in range(len(batch_size)):
                x[j] = np.load(datapath+file_list[int(i*batch_size+j)])
            
            yield x, x[...,1]
        random.shuffle(file_list)
        #print('list_shuffled.')

#==============================================================================
import SimpleITK as sitk
#from skimage.transform import resize
label1_path = r'/home/dkuang/LPBA40/delineation_space/S09/S09.delineation.structure.label.hdr'
label2_path = r'/home/dkuang/LPBA40/delineation_space/S08/S08.delineation.structure.label.hdr'
brain1_path = r'/home/dkuang/LPBA40/delineation_space/S09/S09.delineation.skullstripped.hdr'
brain2_path = r'/home/dkuang/LPBA40/delineation_space/S08/S08.delineation.skullstripped.hdr'
brain1 = sitk.ReadImage(brain1_path)
#brain1 = sitk.Resample(brain1, (res,res,res))
brain1_data = sitk.GetArrayViewFromImage(brain1)
print(brain1_data.shape)
print(np.max(brain1_data))
brain1_data = brain1_data/np.max(brain1_data)
brain2 = sitk.ReadImage(brain2_path)
#brain2 = sitk.Resample(brain2, (res,res,res))
brain2_data = sitk.GetArrayViewFromImage(brain2)
print(brain2_data.shape)
print(np.max(brain2_data))
brain2_data = brain2_data/np.max(brain2_data)


label1 = sitk.ReadImage(label1_path)
label1 = sitk.Resample(label1, (res,res,res))
label1_data = sitk.GetArrayViewFromImage(label1)
print(np.max(label1_data))

label2 = sitk.ReadImage(label2_path)
label2 = sitk.Resample(label2, (res,res,res))
label2_data = sitk.GetArrayViewFromImage(label2)
print(np.max(label2_data))


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
from keras.layers import Input, Conv3D, MaxPooling3D,  UpSampling3D, multiply, Conv3DTranspose, BatchNormalization
from keras.optimizers import Adam
from keras.losses import binary_crossentropy, kullback_leibler_divergence
import keras.backend as K
from keras.regularizers import l1, l2

from mpl_toolkits.mplot3d import Axes3D

#from Utils import vis_grid_3d
    
def total_variation(y):
#    assert K.ndim(y) == 4
    a = K.square(y[:, :res - 1, :res - 1, :res-1, :] - y[:, 1:, :res - 1, :res-1, :])
    b = K.square(y[:, :res - 1, :res - 1, :res-1, :] - y[:, :res - 1, 1:, :res-1, :])
    c = K.square(y[:, :res - 1, :res - 1, :res-1, :] - y[:, :res - 1, :res-1, 1:, :])
    
    return K.pow(K.sum(a + b + c), 0.5)# tweak the power?

def total_variation_loss(yTrue, yPred):
#    assert K.ndim(yTrue) == 4
    diff = yTrue - yPred

    return 10*total_variation(diff) + 0.1*K.pow(K.sum(K.pow(diff, 2)),0.5)


def customloss(yTrue, yPred):
     sse = K.sum(K.square(yTrue - yPred))
     
     Dx_yTrue = yTrue[:, :res - 1, :res - 1, :res-1, :] - yTrue[:, 1:, :res - 1, :res-1, :]
     Dy_yTrue = yTrue[:, :res - 1, :res - 1, :res-1, :] - yTrue[:, :res - 1, 1:, :res-1, :]
     Dz_yTrue = yTrue[:, :res - 1, :res - 1, :res-1, :] - yTrue[:, :res - 1, :res-1, 1:, :]
     
     Dx_yPred = yPred[:, :res - 1, :res - 1, :res-1, :] - yPred[:, 1:, :res - 1, :res-1, :]
     Dy_yPred = yPred[:, :res - 1, :res - 1, :res-1, :] - yPred[:, :res - 1, 1:, :res-1, :]
     Dz_yPred = yPred[:, :res - 1, :res - 1, :res-1, :] - yPred[:, :res - 1, :res-1, 1:, :]
     
     D1 = K.sum(K.square(Dx_yTrue - Dx_yPred))
     D2 = K.sum(K.square(Dy_yTrue - Dy_yPred))
     D3 = K.sum(K.square(Dz_yTrue - Dz_yPred))
     
     return sse+ 0.2*(D1+D2+D3)

input_shape = (res,res,res,2)
def SDN(inputs):
    
    zz = Conv3D(64, (2,2,2), padding = 'same')(inputs)
    zzz = Conv3D(64, (2,2,2), padding = 'same')(zz)
    
    zzz = MaxPooling3D((2,2,2))(zzz)
    zzz = Conv3D(128, (2,2,2), padding = 'same')(zzz)
    
    zzz = UpSampling3D((2,2,2))(zzz) 
    zzz = Conv3D(64, (2,2,2), padding = 'same')(zzz)
    
    zzzz = multiply([zz, zzz])   # easy to cause trouble when shape does not contain enough power of 2.
    zzzz = Conv3D(3, (2,2,2), padding = 'same',
                      kernel_initializer= 'he_normal',
                      bias_initializer = 'he_normal',
#                      activity_regularizer = l2(0.1),
                      activation = 'tanh')(zzz)
    
    locnet = Model(inputs, zzzz)
     
    x1 = SpatialDeformer3D(localization_net=locnet,
                             output_size=(input_shape[0],input_shape[1], input_shape[2]), 
                             input_shape=input_shape)(inputs)
    
    return x1,locnet(inputs)

def SDN_deeper(inputs): # need a deeper one for 3d? one with no pooling?
    z1_1 = Conv3D(32, (2,2,2), padding = 'same')(inputs)
    z1_2 = Conv3D(32, (2,2,2), padding = 'same')(z1_1)
    
    z2 = MaxPooling3D((2,2,2))(z1_2)
    z2_1 = Conv3D(64, (2,2,2), padding = 'same')(z2)
    z2_2 = Conv3D(64, (2,2,2), padding = 'same')(z2_1)
    
    z3 = MaxPooling3D((2,2,2))(z2_2)
    z3 = Conv3D(128, (2,2,2), padding = 'same')(z3)

    
    z3 = UpSampling3D((2,2,2))(z3) 
    z3 = Conv3D(64, (2,2,2), padding = 'same')(z3) # help to overcome local minimum?
#    z3 = Conv3DTranspose(64, (2,2,2), strides=(2,2,2), padding = 'same')(z3)
#    z4 = multiply([z2_1, z3]) 
    
    z4 = UpSampling3D((2,2,2))(z3)
    z5 = Conv3D(32, (2,2,2), padding = 'same')(z4)
#    z5 = Conv3DTranspose(32, (2,2,2), strides=(2,2,2), padding = 'same')(z3)
#    z5= multiply([z1_2, z4])    
    
 
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

def SDN_ver1(inputs): #should control the size carefully, larger strides to downsample 
    z1_1 = Conv3D(32, (2,2,2), padding = 'same')(inputs)
    z1_2 = Conv3D(32, (2,2,2), strides = 2, padding = 'valid')(z1_1)
    z1_2 = BatchNormalization()(z1_2)

    z2_1 = Conv3D(64, (2,2,2), padding = 'same')(z1_2)
    z2_2 = Conv3D(64, (2,2,2), strides = 2, padding = 'valid')(z2_1)
    

    z3 = Conv3D(128, (2,2,2), padding = 'same')(z2_2)

    
    z4 = Conv3DTranspose(64, (2,2,2), strides=(2,2,2), padding = 'valid')(z3)
#    z4 = Conv3D(64, (2,2,2), padding = 'same')(z4)

    z5 = Conv3DTranspose(32, (2,2,2), strides=(2,2,2), padding = 'valid')(z4)   
#    z5 = Conv3D(32, (2,2,2), padding = 'same', activation = 'linear')(z5)
 
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
    
sdn.compile(loss = ['mse', total_variation_loss],
            loss_weights = [1.0, 0e-3],
            optimizer = 'adam' )

x_train = np.expand_dims(np.stack([brain1_data, brain2_data], axis = 3), 0)
y_train = x_train[...,1]
y_train = np.expand_dims(y_train, 4)

epochs = 4
batch_size = 1
#import os
#datapath = r'/home/dkuang/LPBA40_npy/'
#datalist = os.listdir(datapath)
#print(datalist)
#random.shuffle(datalist)
#train_list = datalist[:1]
#print(train_list)
#gen_train = vol_generator(datapath, train_list, batch_size)
#gen_test = vol_generator(test_list, 4)
#history = sdn.fit_generator(gen_train, steps_per_epoch = 1, epochs = epochs, use_multiprocessing = True, verbose=1)

#print(history.history)
#x_train = np.load(datapath+train_list[0])
#print(train_list)
#x_train = np.expand_dims(x_train, 0)
#y_train = x_train[...,1]
#y_train = np.expand_dims(y_train, 4)
#print(x_train.shape)
#print(y_train.shape)
history = sdn.fit(x_train, [y_train, np.zeros([len(y_train), res, res, res, 3])],
            epochs = epochs, batch_size = batch_size,
            verbose = 1, shuffle = True)

# visualize
#plt.figure()
#plt.plot(history.history['loss'])
#warped_vol, deformation = sdn.predict(x_train)

#fig = plt.figure()
#ax = fig.gca(projection='3d')
#color = np.stack((warped_vol[0,...,0],warped_vol[0,...,0],warped_vol[0,...,0],warped_vol[0,...,0]**0.5),axis=3)
#color_mask = color>0.1
#ax.voxels(warped_vol[0,...,0],facecolors=np.clip(color,0,1), edgecolor='k', linestyle = '--',
#          linewidth=0.1)
#vis_grid_3d(deformation[0])


#error_before = (brain1_data-brain2_data)**2
#error_after = (warped_vol[0,...,0]-brain2_data)**2

#from BrainSDN_regpy import j_score
from sklearn.metrics import jaccard_similarity_score
def j_score(yTrue, yPred):
     js=[]
     for yT, yP in zip(yTrue, yPred):
         js.append(jaccard_similarity_score((yT>0).flatten(), (yP>0).flatten()))
     js = np.stack(js)
     return np.mean(js)

# test on one brain

brain_test1 = x_train
warped, deformation = sdn.predict(brain_test1)
#brain_test1 = np.load(datapath+train_list[0])
#print("Brain pair: {}".format(datalist[101]))
print("Before registration: {}".format(j_score(brain_test1[...,1], brain_test1[...,0])))
print("After registration: {}".format(j_score(brain_test1[...,1], warped)))

def Dice(y_true, y_pred):
     T = (y_true.flatten()>0)
     P = (y_pred.flatten()>0)

     return 2*np.sum(T*P)/(np.sum(T) + np.sum(P))

label1_data = np.rint(label1_data)
label2_data = np.rint(label2_data)

label_list = list(np.unique(label1_data))[1:]
print(label_list)
dice_before, dice_after = [], []

'''
Different ways to warp the label mask makes huge differences...
'''
#brain_test1 = np.expand_dims(brain_test1, 0)

for i in label_list:
     dice_before.append(Dice(label2_data==i, label1_data == i))
     seg = K.eval(sdn.layers[-2]._transform(
           deformation, np.expand_dims(
                     np.expand_dims(label1_data==i, 3),0), (res, res, res)))

     dice_after.append(Dice(label2_data == i, np.rint(seg)>0))

#print(j_score(Y_test, sdn.predict(X_test)[0]))

# write to file
#test_log = open('/home/dkuang/SDN3dTest.txt', 'w')
#test_log.write("paramters:")
#test_log.write("*"*40)
#test_log.write("training history:\n")
#for item in history.history['loss']:
#    test_log.write("%s\n"% item)
#test_log.write("Brain pair: {}".format(datalist[101]))
#test_log.write("Before registration: {}".format(j_score(brain_test1[...,1], brain_test1[...,0])))
#brain_test1 = np.expand_dims(brain_test1, 0)
#test_log.write("After registration: {}".format(j_score(brain_test1[...,1], sdn.predict(brain_test1)[0])))
#test_log.close()

#fig = plt.figure()
#ax = fig.gca(projection='3d')
#color = np.stack((np.ones_like(error_before),np.zeros_like(error_before),np.zeros_like(error_before), error_before),axis=3)
#ax.voxels(error_before,facecolors=np.clip(color,0,1), edgecolor='k', linestyle = '--',
#          linewidth=0.05)
#plt.title('Error_before:{0:.2f}'.format(np.sum(error_before)))

#fig = plt.figure()
#ax = fig.gca(projection='3d')
#color = np.stack((np.ones_like(error_after),np.zeros_like(error_after),np.zeros_like(error_after),error_after),axis=3)
#ax.voxels(error_after,facecolors=np.clip(color,0,1), edgecolor='k', linestyle = '--',
#          linewidth=0.05)
#plt.title('Error_after:{0:.2f}'.format(np.sum(error_after)))

'''

'''
#from Utils import multi_slice_viewer
#multi_slice_viewer(warped_vol[0,...,0], 1)
#multi_slice_viewer(deformation[0,...,0], 1) #consider visualizing with a "wild frame"?

# =============================================================================
# test = np.zeros([1,res,res,res,3])
# test_warped = K.eval(sdn.layers[-2]._transform(K.cast(test, dtype = 'float32'), 
#                      np.expand_dims(x_train[...,1],4), 
#                      [res,res,res]))
# #fig = plt.figure()
# #ax = fig.gca(projection='3d')
# #ax.voxels(test_warped[0,...,0],facecolors='b', edgecolor='k')
# #plt.title('0-shift')
# 
# ## test if shift is along z
# #testz = np.zeros_like(test)
# #testz[...,2] = -0.25
# #test_warpedz = K.eval(sdn.layers[-2]._transform(K.cast(testz, dtype = 'float32'), 
# #                     np.expand_dims(x_train[...,1],4), 
# #                     [res,res,res]))
# #
# #fig = plt.figure()
# #ax = fig.gca(projection='3d')
# #ax.voxels(test_warpedz[0,...,0],facecolors='b', edgecolor='k')
# #plt.title('z-shift')
# #
# ## test if shift is along x
# #testx = np.zeros_like(test)
# #testx[...,0] = -0.25
# #test_warpedx = K.eval(sdn.layers[-2]._transform(K.cast(testx, dtype = 'float32'), 
# #                     np.expand_dims(x_train[...,1],4), 
# #                     [res,res,res]))
# #
# #fig = plt.figure()
# #ax = fig.gca(projection='3d')
# #ax.voxels(test_warpedx[0,...,0],facecolors='b', edgecolor='k')
# #plt.title('x-shift')
# #
# ## test if shift is along y
# #testy = np.zeros_like(test)
# #testy[...,1] = -0.25
# #test_warpedy = K.eval(sdn.layers[-2]._transform(K.cast(testy, dtype = 'float32'), 
# #                     np.expand_dims(x_train[...,1],4), 
# #                     [res,res,res]))
# #
# #fig = plt.figure()
# #ax = fig.gca(projection='3d')
# #ax.voxels(test_warpedy[0,...,0],facecolors='b', edgecolor='k')
# #plt.title('y-shift')
# ##plt.ylim(8,0)
# =============================================================================



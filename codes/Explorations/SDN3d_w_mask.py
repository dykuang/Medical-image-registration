# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 16:34:21 2018

@author: Dongyang

This scipt tests the idea of spatially varing regularization strength 

"""
import numpy as np
import random

par = {'res1': 91,

       'res2': 109,

       'res3': 91,

       'kernel_size': (2,2,2),

       'kernel_strides': 2,

       'loss_weights': [1, 2e-5],

       'epochs': 10,

       'batch_size': 16,

       'lr': 1e-3, 

       'w1': 3,

       'w2': 1,

       'w_ctx': 0.2,

       'w_rst': 1, 
       }

print(par)

res1 = par['res1']

res2 = par['res2']

res3 = par['res3']

def vol_generator(path, file_list, batch_size):
    x = np.zeros((batch_size, res1, res2, res3, 2))
    mask = np.ones([batch_size, res1, res2, res3])
    count = 0
    while True:
        for j in range(batch_size):
            temp = np.load(path+file_list[(count*batch_size+j)%len(file_list)])
            x[j] = temp[...,:2]
            mask[j] = temp[...,2]

        yield x, [np.expand_dims(x[...,1],4), np.expand_dims(mask,4)]
       
        count = count + 1
        if count >= len(file_list)/batch_size:
            count = 0

def vol_generator2(path, label_path, file_list, batch_size):
    x = np.zeros((batch_size, res1, res2, res3, 2))
    mask = np.ones([batch_size, res1, res2, res3])
    count = 0
    while True:
        for j in range(batch_size):
            pair = file_list[(count*batch_size+j)%len(file_list)]
            mov = np.load(path+pair[:2]+'.npy')
            fix = np.load(path+pair[2:]+'.npy')
            mov_mask = np.load(label_path + pair[:2]+'.npy')
            mask[j] = ( (mov_mask>0) & (mov_mask<130) ) * 1
            x[j] = np.stack([mov, fix], axis = 3)

        yield x, [np.expand_dims(x[...,1],4), np.expand_dims(mask, 4)]

        count = count + 1
        if count >= len(file_list)/batch_size:
            count = 0

# Training
from spatial_deformer_net3d import SpatialDeformer3D
from keras.models import Model
from keras.layers import Input, Conv3D, MaxPooling3D,  UpSampling3D, multiply, Conv3DTranspose, ZeroPadding3D, Cropping3D, BatchNormalization
from keras.optimizers import Adam
from keras.losses import binary_crossentropy, kullback_leibler_divergence
import keras.backend as K
from keras.regularizers import l1, l2
   
def total_variation(y):
#    assert K.ndim(y) == 4
    a = K.square(y[:, :res1 - 1, :res2 - 1, :res3-1, :] - y[:, 1:, :res2 - 1, :res3-1, :])
    b = K.square(y[:, :res1 - 1, :res2 - 1, :res3-1, :] - y[:, :res1 - 1, 1:, :res3-1, :])
    c = K.square(y[:, :res1 - 1, :res2 - 1, :res3-1, :] - y[:, :res1 - 1, :res2-1, 1:, :])
    
    return K.pow(K.sum(a + b + c), 0.5)# tweak the power?

def total_variation_loss(yTrue, yPred):
#    assert K.ndim(yTrue) == 4
    diff = yPred  # yTrue = 0, ignored.

    return par['w1']*total_variation(diff) + par['w2']*K.pow(K.sum(K.pow(diff, 2)),0.5)

#import tensorflow as tf
def TVL_mask(yTrue, yPred):
    '''
    somewhat unnatural implementation with mask contained in yTrue whereas
    yPred is just the predicted volumn
    '''
    ctx_part = yTrue*yPred # Be aware of the dimension match here
   # rest = tf.logical_not(yTrue)*yPred  # Should yTrue contains two masks instead?
    rest = (1-yTrue)*yPred    

    return par['w_ctx']*total_variation_loss(K.zeros_like(ctx_part),ctx_part) + par['w_rst']*total_variation_loss(K.zeros_like(rest), rest)

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

from keras.layers import PReLU, LeakyReLU
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

inputs = Input(shape = input_shape)
      
sdn = Model(inputs, SDN_ver1(inputs))

#print(sdn.summary())
print(sdn.layers[-1].summary())
#print(sdn.layers)
from keras.optimizers import Adam
sdn.compile(loss = ['mse', TVL_mask],
            loss_weights = par['loss_weights'],
            optimizer = Adam(lr = par['lr'], decay = 1e-5) )


epochs = par['epochs'] 
batch_size = par['batch_size'] 

import os
datapath = r'/LPBA40_npy_flirt/image/'
labelpath = r'/LPBA40_npy_flirt/label/'
validationpath = r'/LPBA40_npy_flirt/image_test/'
testpath = r'/LPBA40_npy_flirt/image_test/'
#datalist = os.listdir(datapath)
#validation_list = os.listdir(validationpath)
#testlist = os.listdir(testpath)

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


gen_train = vol_generator2(datapath, labelpath, train_list, par['batch_size'])
gen_test = vol_generator2(testpath, labelpath, validation_list, par['batch_size'])
history = sdn.fit_generator(gen_train, steps_per_epoch = len(train_list)/par['batch_size'], epochs = par['epochs'], use_multiprocessing = True, verbose=1, validation_data = gen_test, validation_steps = len(validation_list)/par['batch_size'])
loss = history.history['loss']
val_loss = history.history['val_loss']
print("Training complete.")
print("Saving current model ...")
sdn.layers[-1].save_weights('/output/SDN3d_weights.h5') 



from sklearn.metrics import jaccard_similarity_score
def j_score(yTrue, yPred):
     js=[]
     for yT, yP in zip(yTrue, yPred):
          js.append(jaccard_similarity_score((yT>0.1).flatten(), (yP>0.1).flatten()))
     js = np.stack(js)
     return np.mean(js)

# test on one brain
label = '3132.npy' 
#brain_test1 = np.load(testpath+label)

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



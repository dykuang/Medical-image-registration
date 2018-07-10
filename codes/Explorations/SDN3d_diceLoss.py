# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 16:34:21 2018

@author: Dongyang

This script tests using dice score directly 
"""
import numpy as np
import random
par = {'res1': 91,

       'res2': 109,

       'res3': 91,

       'kernel_size': (2,2,2),

       'kernel_strides': 2,

       'loss_weights': [0, 1, 0e-3],

       'epochs': 10,

       'batch_size': 1,

       'lr': 1e-3, 

       'w1': 3,

       'w2': 1 
       }

print(par)

res1 = par['res1']

res2 = par['res2']

res3 = par['res3']

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

def vol_generator3(img_path, label_path, file_list, batch_size):
    x = np.zeros((batch_size, res1, res2, res3, 2))
    xl = np.zeros((batch_size, res1, res2, res3, 2))
    zeros = np.zeros([batch_size, res1, res2, res3, 3])
    count = 0
    while True:
        for j in range(batch_size):
            pair = file_list[(count*batch_size+j)%len(file_list)]
            mov = np.load(img_path+pair[:2]+'.npy')
            fix = np.load(img_path+pair[2:]+'.npy')
            mov_l = np.load(label_path+pair[:2]+'.npy')
            fix_l = np.load(label_path+pair[2:]+'.npy')
#            mov_l = (mov_l > 20) & (mov_l < 35)
#            fix_l = (fix_l > 20) & (fix_l < 35)  
            x[j] = np.stack([mov, fix], axis = 3)
            xl[j] = np.stack([mov_l, fix_l], axis = 3)

        yield [x, xl], [np.expand_dims(x[...,1],4), np.expand_dims(xl[...,1],4), zeros]

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

label_list = [i for i in range(21,35,1)] + [i for i in range(41,51,1)] + [i for i in range(61, 69, 1)] + [i for i in range(81,93,1)] + [101, 102, 121, 122] + [i for i in range(161, 167, 1)] + [181, 182]


def diceLoss(y_true, y_pred):
    top = 2*tf.reduce_sum(y_true * y_pred, [1, 2, 3])
    bottom = tf.maximum(tf.reduce_sum(y_true+y_pred, [1, 2, 3]), 0)
    dice = tf.reduce_mean(top/bottom)
    #return 1/tf.reduce_mean(dice)-1
    return dice

def diceLoss_iter(y_true, y_pred):
    label = tf.convert_to_tensor(label_list, dtype = tf.float32)
    j = tf.constant(0)
    dice = tf.constant(0, dtype = tf.float32)
    c = lambda j, dice: j < 56
    b = lambda j, dice: [j+1, dice+diceLoss(tf.cast(tf.equal(y_true, label[j]), tf.float32) , tf.cast(tf.equal(y_pred,label[j]),tf.float32) )]
    r = tf.while_loop(c,b,[j, dice])
   
    return 1/tf.reduce_mean(r[1])-1 
    
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

from architecture import SDN_ver1 as SDN
#from keras.layers import GaussianNoise
inputs = Input(shape = input_shape)
inputs_l = Input(shape = input_shape)
#aug = GaussianNoise(0.05)(inputs)
disp = SDN(inputs)
disp_M = Model(inputs, disp)
transformer = SpatialDeformer3D(localization_net=disp_M, output_size=(input_shape[0],input_shape[1], input_shape[2]), input_shape=input_shape)

transformer_l = SpatialDeformer3D(localization_net=disp_M, output_size=(input_shape[0],input_shape[1], input_shape[2]), input_shape=input_shape)

warped = transformer(inputs)
warped_l = transformer_l(inputs_l) #really should use a nearest neibor interpolation
#warped = transformer(inputs)

sdn = Model([inputs, inputs_l], [warped, warped_l, disp])
#warped_l = SpatialDeformer3D(localization_net=disp_M, output_size=(input_shape[0],input_shape[1], input_shape[2]), input_shape=input_shape)(inputs_l)
#sdn_pre = Model(inputs, [warped, disp])
#print(sdn_pre.layers[-2][-2])

#warped_l = sdn_pre.layers[-2][-2]._transform(disp, inputs_l[...,0], (91, 109, 91))
#print(sdn.summary())
#print(sdn.output)
#print(sdn.layers[-1].summary())
#print(sdn.layers)
from keras.optimizers import Adam
sdn.compile(loss = [cc3D(), diceLoss_iter, total_variation_loss],
            loss_weights = par['loss_weights'], 
            optimizer = Adam(lr = par['lr'], decay = 1e-5))

epochs = par['epochs'] 
batch_size = par['batch_size'] 

import os
base_dir = r'/home/dkuang/'
datapath = r'/LPBA40_npy/image/'
labelpath =r'/LPBA40_npy/label/'
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

train_list = ['2018']
validation_list = ['2018']
testpath = datapath

gen_train = vol_generator3(datapath, labelpath, train_list, batch_size)
gen_test = vol_generator3(testpath, labelpath, validation_list, batch_size)
history = sdn.fit_generator(gen_train, steps_per_epoch = len(train_list)/batch_size, epochs = epochs, use_multiprocessing = True, verbose=1, validation_data = gen_test, validation_steps = len(validation_list)/batch_size)
loss = history.history['loss']
val_loss = history.history['val_loss']
print(history.history.keys())
print("Training complete.")
print("Saving current model ...")
disp_M.save_weights('/output/SDN3d_weights.h5') 


# test on one brain
label = '2018.npy' 

brain_test1_mov = np.load(testpath+label[:2]+'.npy')
brain_test1_fix = np.load(testpath+label[2:4]+'.npy')
brain_test1 = np.stack([brain_test1_mov, brain_test1_fix], 3)


brain_test_label1 = np.load(labelpath+label[:2]+'.npy')
brain_test_label2 = np.load(labelpath+label[2:4]+'.npy')
brain_test_label = np.stack([brain_test_label1, brain_test_label2], 3)


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
brain_test_label = np.expand_dims(brain_test_label, 0)

warped_brain, warped_label, deformation = sdn.predict([brain_test1, brain_test_label])

from Utils import transform
for i in label_list:

     dice_before.append(Dice(label2_data==i, label1_data == i))
     seg = transform(deformation, np.expand_dims(np.expand_dims(label1_data==i, 3),0), (res1, res2, res3))  

     dice_after.append(Dice(label2_data == i, np.rint(seg)))
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


test_log.write("Before registration: {}\n".format(Dice(brain_test1[...,1], brain_test1[...,0])))
test_log.write("After registration: {}\n".format(Dice(brain_test1[...,1], warped_brain)))
test_log.write("*"*40+"\n")
test_log.write("Before Registration"+"\t" + "After Registration\n")
for before, after in zip(dice_before, dice_after):
    test_log.write("{0:.6f}".format(before)+"\t" + "{0:.6f}".format(after)+'\n')

test_log.close()



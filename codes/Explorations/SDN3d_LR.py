# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 16:34:21 2018

@author: Dongyang

This script tests the regularization with different weights on the 
symmetric and antisymmetric part of the displacement field.
"""
import numpy as np
import random
import Utils1 as U1

par = {'res1': 91,

       'res2': 109,

       'res3': 91,

       'kernel_size': (2,2,2),

       'kernel_strides': 2,

       'loss_weights': [1, 2e-3],

       'epochs': 10,

       'batch_size': 16,

       'lr': 1e-3, 

       'w1': 3,

       'w2': 1,

       'ws': 0.5,

       'wa': 2 
       }

print(par)

res1 = par['res1']

res2 = par['res2']

res3 = par['res3']

def vol_generator2(path, LRpath, file_list, batch_size):
    x = np.zeros((batch_size, res1, res2, res3, 2))
    LR = np.zeros([batch_size, res1, res2, res3, 3])
    count = 0
    while True:
        for j in range(batch_size):
            pair = file_list[(count*batch_size+j)%len(file_list)]
            mov = np.load(path+pair[:2]+'.npy')
            fix = np.load(path+pair[2:]+'.npy')
            x[j] = np.stack([mov, fix], axis = 3)
            LR[j] = np.load(LRpath + pair[:2] +'LR.npy')
            
        yield x, [np.expand_dims(x[...,1],4), LR]

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
    #RES1 = tf.shape(y)[1]
    #RES2 = tf.shape(y)[2]
    #RES3 = tf.shape(y)[3]
#    assert K.ndim(y) == 4
    a = K.square(y[:, :-1, :-1, :-1, :] - y[:, 1:, :- 1, :-1, :])
    b = K.square(y[:, :-1, :-1, :-1, :] - y[:, :-1, 1:, :-1, :])
    c = K.square(y[:, :-1, :-1, :-1, :] - y[:, :-1, :-1, 1:, :])
    
    return K.pow(K.sum(a + b + c), 0.5)# tweak the power?

def total_variation_loss(yTrue, yPred):
#    assert K.ndim(yTrue) == 4
#    diff = yTrue - yPred

    return par['w1']*total_variation(yPred) + par['w2']*K.pow(K.sum(K.pow(yPred, 2)),0.5)

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

def sym_loss(w_s, w_a):
    def myloss(yTrue, yPred):
      #yPred_r = tf.reverse(yPred, 3)
      #yPred_r = yPred[...,::-1,:]
      yTrue_f = tf.reshape(yTrue, (par['batch_size'], res1*res2*res3, 3))
      yTrue_f = tf.transpose(yTrue_f, (0, 2, 1))
      x_s = tf.slice(yTrue_f, [0, 0, 0], [-1, 1, -1]) #problem here?
      y_s = tf.slice(yTrue_f, [0, 1, 0], [-1, 1, -1])
      z_s = tf.slice(yTrue_f, [0, 2, 0], [-1, 1, -1])
      X = tf.reshape(x_s, [-1])
      Y = tf.reshape(y_s, [-1])
      Z = tf.reshape(z_s, [-1])  
      yPred_r = U1.interpolate(yPred, Y, Z, X, (res1,res2,res3))
      yPred_r = tf.reshape(yPred_r, (par['batch_size'], res1, res2, res3, 3))
      
      #Dh = U1.Jac(yTrue)
      Dh = U1.Jac_5(yTrue)
      yPred_pf = tf.einsum('abcdef,abcdf->abcde', Dh, yPred_r[:,2:-2,2:-2,2:-2,:]) # change the subset accordingly with choices of jacobian estimation

      y_s = 0.5*(yPred[:,:-1,:-1,:-1,:] + yPred_pf)
      y_a = 0.5*(yPred[:,:-1,:-1,:-1,:] - yPred_pf)
      #yTrue = tf.reshape(yTrue, (-1, res1, res2, res3, 3))
      return w_s*total_variation_loss(yTrue, y_s) + w_a*total_variation_loss(yTrue, y_a)
    return myloss

from architecture import SDN_ver1 as SDN
#from keras.layers import GaussianNoise
inputs = Input(shape = input_shape)
#aug = GaussianNoise(0.05)(inputs)
#disp = SDN(inputs)
disp_M = Model(inputs, SDN(inputs))
warped = SpatialDeformer3D(localization_net=disp_M, output_size=(input_shape[0],input_shape[1], input_shape[2]), input_shape=input_shape)(inputs)

sdn = Model(inputs, [warped, disp_M(inputs)])
   
#print(sdn.summary())
print(sdn.layers[-1].summary())
#print(sdn.layers)
from keras.optimizers import Adam
sdn.compile(loss = [cc3D(), sym_loss(par['ws'], par['wa'])],
            loss_weights = par['loss_weights'],
            optimizer = Adam(lr = par['lr'], decay = 1e-5) )

#x_train = np.expand_dims(np.stack([brain1_data, brain2_data], axis = 3), 0)
#y_train = x_train[...,1]
#y_train = np.expand_dims(y_train, 4)

epochs = par['epochs'] 
batch_size = par['batch_size'] 

import os
datapath = r'/LPBA40_npy/image/'
labelpath = r'/LPBA40_npy/label/'
testpath = r'/LPBA40_npy/image_test/'
LRpath = r'/LPBA40_LR/'
#datalist = os.listdir(datapath)
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

#datapath = testpath
#train_list = validation_list
#train_list = ['3239']
#validation_list = ['3239']
'''
Training stucks at providing data without going to actual training. Did not know why...

'''
#train_list = train_list[:320]
#validation_list = validation_list[:25]

gen_train = vol_generator2(datapath, LRpath, train_list, batch_size)
gen_test = vol_generator2(testpath, LRpath, validation_list, batch_size)
history = sdn.fit_generator(gen_train, steps_per_epoch = len(train_list)/batch_size, epochs = epochs, use_multiprocessing = True, verbose=1, validation_data = gen_test, validation_steps = len(validation_list)/batch_size)
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
label = '3239.npy' 

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



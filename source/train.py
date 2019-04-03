# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 16:34:21 2018

@author: Dongyang

This script tests spatial_deformer_net3d.py
"""
import numpy as np
import random

par = {'res1': 144,

       'res2': 180,

       'res3': 144,

       'kernel_size': (3,3,3),

       'kernel_strides': 2,

       'loss_weights': [1.0, 1.0],

       'epochs': 10,

       'batch_size': 2,

       'lr': 1e-4, 

       'w1': 1.0,

       'w2': 0.0,

       'cc_size': 9, 
       
       'NJ loss': 1e-3
       }

print(par)

res1 = par['res1']

res2 = par['res2']

res3 = par['res3']

epochs = par['epochs'] 

batch_size = par['batch_size'] 

datapath = r'path/storing/imagepairs'

labelpath = r'path/storing/anotatedlabels'

outpath = r'output/path'

outname = r'output/name'

input_shape = (res1, res2, res3, 2)

def vol_generator2(path, file_list, batch_size):
    '''
    A volume generator to provide mini-batches
    '''
    x = np.zeros((batch_size, res1, res2, res3, 2))
    zeros = np.zeros([batch_size, res1, res2, res3, 3])
    count = 0
    while True:
        for j in range(batch_size):
            pair = file_list[(count*batch_size+j)%len(file_list)]
            mov = np.load(path+pair[:3]+'.npy')
            fix = np.load(path+pair[3:]+'.npy')
            x[j] = np.stack([mov, fix], axis = 3)

        yield x, [np.expand_dims(x[...,1],4), zeros]

        count = count + 1
        if count >= len(file_list)/batch_size:
            count = 0
            random.shuffle(file_list)
            
# Training
from spatial_deformer_net3d import SpatialDeformer3D
from keras.models import Model
from keras.layers import Input
from keras.optimizers import Adam
#from keras.losses import binary_crossentropy, kullback_leibler_divergence
#import keras.backend as K
#from keras.regularizers import l1, l2
#import tensorflow as tf

  
from architecture import SDN_incept as SDN
#from keras.layers import GaussianNoise
inputs = Input(shape = input_shape)
#aug = GaussianNoise(0.05)(inputs)
disp_M = Model(inputs, SDN(inputs))
warped = SpatialDeformer3D(localization_net=disp_M, output_size=(input_shape[0],input_shape[1], input_shape[2]), input_shape=input_shape)(inputs)

sdn = Model(inputs, [warped, disp_M(inputs)])
   
#print(sdn.summary())
print(sdn.layers[-1].summary())
#print(sdn.layers)

from losses import cc3D, gradientLoss, NJ_loss
def reg_loss(y_true, y_pred):
    return gradientLoss('l2')(y_true, y_pred) + par['NJ loss']*NJ_loss(y_true, y_pred)

sdn.compile(loss = [cc3D(win=[par['cc_size'],par['cc_size'],par['cc_size']]), reg_loss],
            loss_weights = par['loss_weights'],
            optimizer = Adam(lr = par['lr'], decay = 1e-4) )

'''
replace the following if you have a different split
'''
train_files = ['{:03d}'.format(i) for i in range(38, 80)] 
val_files =  ['{:03d}'.format(i) for i in range(0, 20)]

train_list = []
validation_list = []
from itertools import combinations
for ind in combinations(range(0,len(train_files),1),2):
    train_list.append(train_files[ind[0]]+train_files[ind[1]])
    train_list.append(train_files[ind[1]]+train_files[ind[0]])
for ind in combinations(range(0,len(val_files),1),2):
    validation_list.append(val_files[ind[0]]+val_files[ind[1]])
    validation_list.append(val_files[ind[1]]+val_files[ind[0]])

gen_train = vol_generator2(datapath, train_list, batch_size)
gen_test = vol_generator2(datapath, validation_list, batch_size)

#from keras.callbacks import ModelCheckpoint
#mc = ModelCheckpoint(outpath+'SDN3d_weights_TVS16_{epoch:02d}.h5', save_weights_only=True, verbose=1, period=5)

history = sdn.fit_generator(gen_train, steps_per_epoch = len(train_list)/batch_size, epochs = epochs, use_multiprocessing = True, verbose=1, validation_data = gen_test, validation_steps = len(validation_list)/batch_size)
loss = history.history['loss']
val_loss = history.history['val_loss']
print("Training complete.")
print("Saving current model ...")
sdn.layers[-1].save_weights(outpath + outname) 
print("Saving model weights to"+outpath + outname)
print(loss)
print('*'*40)
print(val_loss)



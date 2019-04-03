# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 16:34:21 2018

@author: Dongyang

This script tests voxelmorph 

using a different resolution than original
"""
import numpy as np
import random
par = {'res1': 144,

       'res2': 180,

       'res3': 144,

       'loss_weights': [1, 1],

       'epochs': 10,

       'batch_size': 3,

       'lr': 1e-4,
     
       }

print(par)

datapath = ''
output_dir = ''
output_name = ''

res1 = par['res1']

res2 = par['res2']

res3 = par['res3']

def vol_generator2(path, file_list, batch_size):
    src = np.zeros((batch_size, res1, res2, res3,1))
    tgt = np.zeros((batch_size, res1, res2, res3,1))

    zeros = np.zeros([batch_size, res1, res2, res3, 3])
    count = 0
    while True:
        for j in range(batch_size):
            pair = file_list[(count*batch_size+j)%len(file_list)]
            src[j,...,0] = np.load(path+pair[:3]+'.npy')
            tgt[j,...,0] = np.load(path+pair[3:]+'.npy')
            
        yield [src, tgt], [tgt, zeros]

        count = count + 1
        if count > len(file_list)//batch_size:
            count = 0
            random.shuffle(file_list)

# Training
#from spatial_deformer_net3d import SpatialDeformer3D
from keras.models import Model
from keras.layers import Input, Conv3D, MaxPooling3D,  UpSampling3D, multiply, Conv3DTranspose, ZeroPadding3D, Cropping3D, concatenate,BatchNormalization, LeakyReLU, Lambda, PReLU
from keras.optimizers import Adam
import keras.backend as K
from keras.initializers import RandomNormal
import losses
from dense_3D_spatial_transformer import Dense3DSpatialTransformer
def myConv(x_in, nf, strides=1):
    """
    specific convolution module including convolution followed by leakyrelu
    """

    x_out = Conv3D(nf, kernel_size=3, padding='same',
                   kernel_initializer='he_normal', strides=strides)(x_in)
    x_out = LeakyReLU(0.2)(x_out)
    return x_out



def unet(vol_size, enc_nf, dec_nf, full_size=True):
    """
    unet network for voxelmorph 

    Args:
        vol_size: volume size. e.g. (256, 256, 256)
        enc_nf: encoder filters. right now it needs to be to 1x4.
            e.g. [16,32,32,32]
            TODO: make this flexible.
        dec_nf: encoder filters. right now it's forced to be 1x7.
            e.g. [32,32,32,32,8,8,3]
            TODO: make this flexible.
        full_size

    """

    # inputs
    src = Input(shape=(vol_size[0], vol_size[1], vol_size[2],1))
    tgt = Input(shape=(vol_size[0], vol_size[1], vol_size[2],1))

    x_in = concatenate([src, tgt], 4)
    #x_in = ZeroPadding3D(((3,2), (2,1), (3,2)))(x_in)
    print(K.int_shape(x_in))


    # down-sample path.
    x0 = myConv(x_in, enc_nf[0], 2)  
    x1 = myConv(x0, enc_nf[1], 2)  
    x2 = myConv(x1, enc_nf[2], 2)  
    x3 = myConv(x2, enc_nf[3], 2)  

    # up-sample path.
    x = myConv(x3, dec_nf[0])
    x = UpSampling3D()(x)
    x2 = ZeroPadding3D(((0,0), (0,1), (0,0)))(x2)
    x = concatenate([x, x2])
    x = myConv(x, dec_nf[1])
    x = UpSampling3D()(x)
    x1 = ZeroPadding3D(((0,0), (1,2), (0,0)))(x1)
    x = concatenate([x, x1])
    x = myConv(x, dec_nf[2])
    x = UpSampling3D()(x)
    x0 = ZeroPadding3D(((0,0), (3,3), (0,0)))(x0)
    x = concatenate([x, x0])
    x = myConv(x, dec_nf[3])
    x = myConv(x, dec_nf[4])

    if full_size:
        x = UpSampling3D()(x)
        x_in = ZeroPadding3D(((0,0), (6,6), (0,0)))(x_in)
        x = concatenate([x, x_in])
        x = myConv(x, dec_nf[5])

        # optional convolution
        if (len(dec_nf) == 8):
            x = myConv(x, dec_nf[6])
    
    # transform the results into a flow.
    flow = Conv3D(dec_nf[-1], kernel_size=3, padding='same',
                  kernel_initializer=RandomNormal(mean=0.0, stddev=1e-5), name='flow')(x)
    flow = Lambda(lambda x: x[:,:, 6:-6, :, :])(flow)
    # warp the source with the flow
    y = Dense3DSpatialTransformer()([src, flow])

    # prepare model
    model = Model(inputs=[src, tgt], outputs=[y, flow])
    warped_img, forward_flow = model([src, tgt])
    warped_back, backward_flow = model([warped_img, src])
    
    return model, warped_back, backward_flow, src



sdn, warped_back, backward_flow, src = unet((res1, res2, res3), [16,32,32,32], [32,32,32,32,8,8,3])
print(sdn.summary())   

def img_loss():
    def Loss(yTrue, yPred):
        return losses.cc3D()(yTrue, yPred) + losses.cc3D()(warped_back, src)
    return Loss
        
def grad_loss():
    def Loss(yTrue, yPred):
        return losses.gradientLoss('l2')(yTrue, yPred) + losses.gradientLoss('l2')(0,backward_flow)
    return Loss 

  

sdn.compile(loss = [img_loss(), grad_loss()],
            loss_weights = par['loss_weights'], 
#            metrics = [rec_img_loss, reg_grad],
            optimizer = Adam(lr = par['lr'], decay = 1e-5) )


epochs = par['epochs'] 
batch_size = par['batch_size'] 




#train_files = np.load('/global/home/hpc4355/MindBoggle/output/train.npy')
#val_files=np.load('/global/home/hpc4355/MindBoggle/output/val.npy')

NKI_RS = ['{:03d}'.format(i) for i in range(38, 60)]
NKI_TRT = ['{:03d}'.format(i) for i in range(60, 80)]
OASIS_TRT = ['{:03d}'.format(i) for i in range(80, 100)]

train_files = NKI_RS + OASIS_TRT
val_files = NKI_TRT

# if using vol_generator2
train_list = []
validation_list=[]
from itertools import combinations
for ind in combinations(range(0,len(train_files),1),2):
    train_list.append(train_files[ind[0]][:3]+train_files[ind[1]][:3])
    train_list.append(train_files[ind[1]][:3]+train_files[ind[0]][:3])
for ind in combinations(range(0,len(val_files),1),2):
    validation_list.append(val_files[ind[0]][:3]+val_files[ind[1]][:3])
    validation_list.append(val_files[ind[1]][:3]+val_files[ind[0]][:3])

'''
Training stucks at providing data without going to actual training. Did not know why...

'''

gen_train = vol_generator2(datapath, train_list, batch_size)
gen_test = vol_generator2(datapath, validation_list, batch_size)
history = sdn.fit_generator(gen_train, steps_per_epoch = len(train_list)/batch_size, epochs = epochs, use_multiprocessing = True, verbose=1, validation_data = gen_test, validation_steps = len(validation_list)/batch_size)
loss = history.history['loss']
val_loss = history.history['val_loss']
print("Training complete.")
print("Saving current model ...")
sdn.save_weights(output_dir + output_name) 



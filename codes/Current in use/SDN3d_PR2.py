'''
This script tests the idea of prediction + refinement
'''
import numpy as np
from keras.models import Model
from spatial_deformer_net3d import SpatialDeformer3D
from architecture import SDN_ver1, SDN_ver2, SDN_ver3, SDN_ver4, SDN_ver5, SDN_ver6
from keras.layers import Input
from keras.optimizers import Adam
from Utils import Dice, transform
import os
import keras.backend as K
par = {'res1': 91,

       'res2': 109,

       'res3': 91,

       'kernel_size': (2,2,2),

       'kernel_strides': 2,

       'loss_weights': [1, 0e-5],

       'epochs': 5,

       'batch_size': 1,

       'lr': 5e-5,

       'w1': 3,

       'w2': 1 
       }

print(par)

res1 = par['res1']

res2 = par['res2']

res3 = par['res3']


def vol_generator(path, file_list, batch_size):
    x = np.zeros((batch_size, res1, res2, res3, 2))
    zeros = np.zeros([batch_size, res1, res2, res3, 3])
    count = 0
    while True:
        for j in range(batch_size):
            pair = file_list[(count*batch_size+j)%len(file_list)]
            mov = np.load(path+pair[:2]+'.npy')
            fix = np.load(path+pair[2:]+'.npy')
            x[j] = np.stack([mov, fix], axis = 3)

        yield x, [x[...,:1], zeros]

        count = count + 1
        if count >= len(file_list)/batch_size:
            count = 0

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

input_shape = (res1,res2,res3,2)

inputs = Input(shape = input_shape)

dispnet = Model(inputs, SDN_ver1(inputs)) #change this when using different models


datapath = r'/home/dkuang/LPBA40_npy/image/'
labelpath = r'/home/dkuang/LPBA40_npy/label/'
predpath = r'/home/dkuang/'

warped = SpatialDeformer3D(localization_net=dispnet,
                             output_size=(input_shape[0],input_shape[1], input_shape[2]),
                             input_shape=input_shape)(inputs)

sdn_pred = Model(inputs, [warped, dispnet(inputs)])
#print(sdn_pred.summary())
dispnet.load_weights(r'result_test/SDN3d_weights_3_1.h5') # location does not seems to matter

for layer in sdn_pred.layers:
    layer.trainable = False

def vol_gen_refine(path, pred_path, file_list, batch_size):
    x = np.zeros((batch_size, res1, res2, res3, 2))
    zeros = np.zeros([batch_size, res1, res2, res3, 3])
    count = 0
    while True:
        for j in range(batch_size):
            pair = file_list[(count*batch_size+j)%len(file_list)]
            mov = np.load(pred_path+pair+'.npy')
#            mov = mov/np.max(mov)
            fix = np.load(path+pair[2:]+'.npy')
            x[j] = np.stack([mov, fix], axis = 3)           
 
        yield x, [np.expand_dims(x[...,1],4), zeros]
        
        count = count + 1
        if count >= len(file_list)/batch_size:
            count = 0

inputs_refine = Input(shape = input_shape)
dispnet_refine = Model(inputs_refine, SDN_ver1(inputs_refine))
warped_refine = SpatialDeformer3D(localization_net=dispnet_refine,
                             output_size=(input_shape[0],input_shape[1], input_shape[2]),
                             input_shape=input_shape)(inputs_refine)

sdn_refine = Model(inputs_refine, [warped_refine, dispnet_refine(inputs_refine)])

print(sdn_refine.summary())
label_list = [i for i in range(21,35,1)] + [i for i in range(41,51,1)] + [i for i in range(61, 69, 1)] + [i for i in range(81,93,1)] + [101, 102, 121, 122] + [i for i in range(161, 167, 1)] + [181, 182]

train_list = []
from itertools import combinations
for ind in combinations(range(31,41,1),2):
    train_list.append('{0:02d}{1:02d}'.format(ind[0],ind[1]))
    train_list.append('{0:02d}{1:02d}'.format(ind[1],ind[0]))

train_list = ['2016']
print(train_list)

dice_before = np.zeros([len(train_list), 56])
for test_ind, test_label in enumerate(train_list):
    brain_test_mov = np.load(datapath+test_label[:2]+'.npy')
    brain_test_fix = np.load(datapath+test_label[2:]+'.npy')
    brain_test = np.stack([brain_test_mov, brain_test_fix], 3)
   

    brain_test_label1 = np.load(labelpath+test_label[:2]+'.npy')
    brain_test_label2 = np.load(labelpath+test_label[2:4]+'.npy')

    brain_test = np.expand_dims(brain_test, 0)
    warped_vol, deformation = sdn_pred.predict(brain_test)
    np.save('/home/dkuang/{}.npy'.format(test_label), warped_vol[0,...,0])

    for label_ind, i in enumerate(label_list):
        seg = transform(deformation, np.expand_dims(np.expand_dims(brain_test_label1==i, 3),0), (res1, res2, res3))

        dice_before[test_ind, label_ind] = Dice(brain_test_label2 == i, np.rint(seg)>0)
 
print(dice_before)
print(np.mean(dice_before, 1))

sdn_refine.compile(loss = ['mse', total_variation_loss],
            loss_weights = par['loss_weights'],
            optimizer = Adam(lr = par['lr'], decay = 1e-5) )
print(sdn_refine.summary())

gen_train = vol_gen_refine(datapath, predpath,train_list, par['batch_size'])
history = sdn_refine.fit_generator(gen_train, steps_per_epoch = len(train_list)/par['batch_size'], epochs = par['epochs'], use_multiprocessing = True, verbose=1)


dice_after = np.zeros([len(train_list), 56])
for test_ind, test_label in enumerate(train_list):
    brain_test_mov = np.load(datapath+test_label[:2]+'.npy')
    brain_test_fix = np.load(datapath+test_label[2:]+'.npy')
    brain_test = np.stack([brain_test_mov, brain_test_fix], 3)

    brain_test_label1 = np.load(labelpath+test_label[:2]+'.npy')
    brain_test_label2 = np.load(labelpath+test_label[2:4]+'.npy')

    brain_test = np.expand_dims(brain_test, 0)

    brain_test_pred, deformation_pred = sdn_pred.predict(brain_test)
    #brain_test_pred = brain_test_pred/np.max(brain_test_pred)
    #print(brain_test_pred.shape)
    #print(brain_test[...,1:].shape)
 
    deformation_refine = dispnet_refine.predict(np.stack([brain_test_pred,brain_test[..., 1:]],4)[...,0])
   # deformation_refine = sdn_refine.layers[-1].predict(np.stack([brain_test_pred,brain_test[..., 1:]],4)[...,0])

    for label_ind, i in enumerate(label_list):
        seg = transform(deformation_pred, np.expand_dims(np.expand_dims(brain_test_label1==i, 3),0), (res1, res2, res3))
#        seg = np.rint(seg)>0
        seg_refine = transform(deformation_refine, seg, (res1, res2, res3))

        dice_after[test_ind, label_ind] = Dice(brain_test_label2 == i, np.rint(seg_refine)>0)
        #print("ROI {} result appended..".format(i))

    print('Test sample {}\'s evaluation completed.'.format(test_ind+1))

print(dice_after)
print(np.mean(dice_after, 1))
#np.save('result_test/dice_after_PR_3_1.npy', dice_after)


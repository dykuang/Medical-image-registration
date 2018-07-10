'''
This script tests the idea of prediction + refinement
'''
import numpy as np
from keras.models import Model
from spatial_deformer_net3d import SpatialDeformer3D
from architecture import SDN_ver1 
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

       'loss_weights': [1, 2e-5],

       'epochs': 5,

       'batch_size': 16,

       'lr': 5e-4,

       'w1': 0.5,

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

        yield x, [np.expand_dims(x[...,1],4), zeros]

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


datapath = r'/LPBA40_npy/image_test/'
labelpath = r'/LPBA40_npy/label/'


warped = SpatialDeformer3D(localization_net=dispnet,
                             output_size=(input_shape[0],input_shape[1], input_shape[2]),
                             input_shape=input_shape)(inputs)

sdn = Model(inputs, [warped, dispnet(inputs)])

sdn.compile(loss = ['mse', total_variation_loss],
            loss_weights = par['loss_weights'],
            optimizer = Adam(lr = par['lr'], decay = 1e-5) )
print(sdn.summary())
print('Saving the fine_tuned model..')
sdn.layers[-1].load_weights(r'/weights/SDN3d_weights.h5')
print('Saving completed..')

train_list = []
from itertools import combinations
for ind in combinations(range(31,41,1),2):
    train_list.append('{0:02d}{1:02d}'.format(ind[0],ind[1]))
    train_list.append('{0:02d}{1:02d}'.format(ind[1],ind[0]))

#train_list = ['3132']

gen_train = vol_generator(datapath, train_list, par['batch_size'])
history = sdn.fit_generator(gen_train, steps_per_epoch = len(train_list)/par['batch_size'], epochs = par['epochs'], use_multiprocessing = True, verbose=1)

label_list = [i for i in range(21,35,1)] + [i for i in range(41,51,1)] + [i for i in range(61, 69, 1)] + [i for i in range(81,93,1)] + [101, 102, 121, 122] + [i for i in range(161, 167, 1)] + [181, 182]


dice_after = np.zeros([len(train_list), 56])
for test_ind, test_label in enumerate(train_list):
    brain_test_mov = np.load(datapath+test_label[:2]+'.npy')
    brain_test_fix = np.load(datapath+test_label[2:]+'.npy')
    brain_test = np.stack([brain_test_mov, brain_test_fix], 3)

    brain_test_label1 = np.load(labelpath+test_label[:2]+'.npy')
    brain_test_label2 = np.load(labelpath+test_label[2:4]+'.npy')

    brain_test = np.expand_dims(brain_test, 0)
   # deformation = sdn.predict(brain_test)[1]
    deformation = sdn.layers[-1].predict(brain_test)
   # deformation = dispnet.predict(brain_test)
    for label_ind, i in enumerate(label_list):
        seg = transform(deformation, np.expand_dims(np.expand_dims(brain_test_label1==i, 3),0), (res1, res2, res3))

        dice_after[test_ind, label_ind] = Dice(brain_test_label2 == i, np.rint(seg)>0)
        #print("ROI {} result appended..".format(i))
    print('Test sample {}\'s evaluation completed.'.format(test_ind+1))


np.save('/output/dice_after_PR.npy', dice_after)


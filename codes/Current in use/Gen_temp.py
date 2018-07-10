'''
This script tests the idea for generating average templates
'''

import numpy as np
from keras.models import Model
#from spatial_deformer_net3d import SpatialDeformer3D
from architecture import SDN_ver11 as SDN
from keras.layers import Input
from Utils import Dice, transform
import os


res1, res2, res3 = 91, 109, 91
input_shape = (res1,res2,res3,2)

inputs = Input(shape = input_shape)

sdn = Model(inputs, SDN(inputs)) #change this when using different models

print(sdn.summary())

datapath = r'/home/dkuang/LPBA40_npy/image_test/'
sdn.load_weights(r'result_test/SDN3d_weights_243.h5')

print("Loading weights from SDN3d_weights_243.h5")

samples = ['34', '35', '36', '32', '38', '40']

def ini_temp(samples = samples):
    data = 0
    for sample in samples:
      data += np.load(datapath+sample+'.npy')
    
    data = data/len(samples)
    data = data/np.max(data)
    return data

ini = ini_temp()
ini = np.expand_dims(np.expand_dims(ini, 3),0)
#print(ini.shape)

'''
test transform
'''
ini1 = transform(np.zeros([1,res1, res2, res3, 3]), ini, (res1, res2, res3))
print(np.all(ini1==ini))
print([np.min(ini1-ini), np.max(ini1-ini)])

alpha = 0.2
for i in range(20):
    disp_sum = 0
    for sample in samples:
       tgt = np.load(datapath + sample + '.npy')
       tgt = np.expand_dims(np.expand_dims(tgt, 3),0)
       pair = np.stack([ini, tgt], 4)[...,0]
#       print(np.all(pair[0,...,1] == tgt[0,...,0]))
       #print(tgt.shape)
       disp_sum+=sdn.predict(pair)
    disp_sum = disp_sum/len(samples)
    print(np.linalg.norm(disp_sum))
    ini = transform(alpha*disp_sum, ini, (res1, res2, res3))
    #print(ini.shape)


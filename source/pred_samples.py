'''
This script provides some prediction examples
'''
import numpy as np
from keras.models import Model
from spatial_deformer_net3d import SpatialDeformer3D
from architecture import SDN_incept as SDN
from keras.layers import Input
#from Utils import Dice, transform
#import os

res1, res2, res3 = 144, 180, 144

input_shape = (res1,res2,res3,2)

inputs = Input(shape = input_shape)

disp_M = Model(inputs, SDN(inputs)) 

print(disp_M.summary())
source_data = r''
datapath = r''
disp_M.load_weights(r'')

_warped = SpatialDeformer3D(localization_net=disp_M, output_size=(input_shape[0],input_shape[1], input_shape[2]), input_shape=input_shape)(inputs)

sdn = Model(inputs, [ _warped, disp_M(inputs) ] )

test_files = ['{:03d}.npy'.format(i) for i in range(80, 100)]
test_list = []

from itertools import combinations
for ind in combinations(range(0,len(test_files),1),2):
    test_list.append(test_files[ind[0]][:3]+test_files[ind[1]][:3])
    test_list.append(test_files[ind[1]][:3]+test_files[ind[0]][:3])

from Utils import Get_Ja
for sample in test_list[:3]:
    mov = np.load(source_data+sample[:3]+'.npy')
    fix = np.load(source_data+sample[3:]+'.npy')
    
    pair = np.stack([mov, fix], axis=3)
    pair = np.expand_dims(pair, 0)
    warped, disp = sdn.predict(pair)
    
#    np.save('result_test/S'+sample_pair[:2]+'.npy', mov)
#    np.save('result_test/S'+sample_pair[2:]+'.npy', fix)
    np.save(''+sample+'.npy', warped[0,...,0])
    np.save(''+sample+'.npy', Get_Ja(disp))

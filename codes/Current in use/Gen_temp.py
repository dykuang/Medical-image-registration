'''
This script tests the idea for generating average templates
'''

import numpy as np
from keras.models import Model
from spatial_deformer_net3d import SpatialDeformer3D
from architecture import SDN_ver11 as SDN
from keras.layers import Input
from Utils import Dice, transform
import os


res1, res2, res3 = 91, 109, 91
input_shape = (res1,res2,res3,2)

inputs = Input(shape = input_shape)

disp_M = Model(inputs, SDN(inputs)) #change this when using different models

warped = SpatialDeformer3D(localization_net=disp_M, output_size=(input_shape[0],input_shape[1], input_shape[2]), input_shape=input_shape)(inputs)

sdn = Model(inputs, [warped, disp_M(inputs)])

print(sdn.summary())

datapath = r'/home/dkuang/LPBA40_npy/image_test/'
weights_path = r'/home/dkuang/Github/Medical-image-registration/'
disp_M.load_weights(weights_path+r'result_test/SDN3d_weights_243.h5')

print("Loading weights from SDN3d_weights_243.h5")

#samples = ['34', '35', '36', '32', '38', '40']
samples = os.listdir(datapath)

def ini_temp(samples = samples):
    data = 0
    for sample in samples:
      data += np.load(datapath+sample)
    
    data = data/len(samples)
    data = data/np.max(data)
    return data

ini = ini_temp()
#np.save(weights_path+r'result_test/avg_direct.npy', ini)

#import SimpleITK as sitk
#avg = sitk.ReadImage(r'/home/dkuang/LPBA40/avg152T1_brain.nii')
#avg_data = sitk.GetArrayViewFromImage(avg) 
#ini = avg_data/np.max(avg_data)

ini = np.expand_dims(np.expand_dims(ini, 3),0)
#print(ini.shape)

'''
test transform
'''
ini1 = transform(np.zeros([1,res1, res2, res3, 3]), ini, (res1, res2, res3))
print(np.all(ini1==ini))
print([np.min(ini1-ini), np.max(ini1-ini)])

#alpha = 0.5

#for i in range(20):
#    disp_sum = 0
#    for sample in samples:
#       tgt = np.load(datapath + sample)
#       tgt = np.expand_dims(np.expand_dims(tgt, 3),0)
#       pair = np.stack([ini, tgt], 4)[...,0]
#       print(np.all(pair[0,...,1] == tgt[0,...,0]))
       #print(tgt.shape)
#       disp_sum+=sdn.predict(pair)
#    disp_sum = disp_sum/len(samples)
#    print(np.linalg.norm(disp_sum))
#    ini = transform(alpha*disp_sum, ini, (res1, res2, res3))
#    ini = ini/np.max(ini)
    #print(ini.shape)

'''
idea 1

only update the center

equal weight average
'''
#epsilon = 11 
#diff = 100
#while diff > epsilon:
#    warped = np.zeros_like(ini)
#    for sample in samples:
#       tgt = np.load(datapath + sample)
#       tgt = np.expand_dims(np.expand_dims(tgt, 3),0)
#       pair = np.stack([tgt, ini], 4)[...,0] #warp towards current guess 
#
#       warped += sdn.predict(pair)[0]
#    ini_p = warped/len(samples)
#    ini_p = ini_p/np.max(ini_p)
#    diff = np.linalg.norm(ini-ini_p)  # change a criterion?
#    print(diff)
#    ini = ini_p

'''
idea 2

only update the center

weighted average by |D_phi|
'''
from Utils import Get_Ja
epsilon = 2e-3
diff = 100
while diff > epsilon:
    warped = np.zeros_like(ini)
    D_sum = 0
    for sample in samples:
       tgt = np.load(datapath + sample)
       tgt = np.expand_dims(np.expand_dims(tgt, 3),0)
       pair = np.stack([tgt, ini], 4)[...,0] #warp towards current guess 
       
       img, disp = sdn.predict(pair)
       D_phi = Get_Ja(disp)
       warped += D_phi*img
       D_sum += D_phi
       
    ini_p = warped/D_sum
    ini_p = ini_p/np.max(ini_p)
    diff = np.mean(np.abs(ini_p-ini))
    print(diff)

    ini = ini_p

'''
idea 3

update center and sample

weighted average
'''
#epsilon = .3
#diff = 100
#tgts = np.zeros([10,91,109,91])
#warped = np.zeros_like(ini)
#for i, sample in enumerate(samples):
#    tgts[i] = np.load(datapath + sample)

#while diff>epsilon:
#    warped = np.zeros_like(ini)
#    D_sum = 0
#    for i in range(10):
#       tgt = np.expand_dims(np.expand_dims(tgts[i], 3),0)
#       pair = np.stack([tgt, ini], 4)[...,0] #warp towards current guess 

#       img, disp = sdn.predict(pair)
#       D_phi = Get_Ja(disp)
#       warped += D_phi*img
#       D_sum += D_phi

#       tgts[i] = img[0,...,0] #also update samples

#   ini_p = warped/D_sum
#    ini_p = ini_p/np.max(ini_p)
#    diff = np.max(np.abs(ini_p-ini))
#    print(diff)
#    ini = ini_p
    



np.save(weights_path+r'result_test/avg_avg08091e-3.npy', ini)

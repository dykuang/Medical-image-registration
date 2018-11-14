'''
This script tests the idea for generating average templates
'''

import numpy as np
from keras.models import Model
from spatial_deformer_net3d import SpatialDeformer3D
from architecture import SDN_incept as SDN
from keras.layers import Input
from Utils import Dice
#import os
#from scipy.ndimage import gaussian_filter
import time

start_time = time.time()

res1, res2, res3 = 144, 180, 144

datapath = r'path/storing/images/'

weights_path = r'path/storing/trained/weights'

weights_name = r''

input_shape = (res1,res2,res3,2)

inputs = Input(shape = input_shape)

disp_M = Model(inputs, SDN(inputs)) #change this when using different models

warped = SpatialDeformer3D(localization_net=disp_M, output_size=(input_shape[0],input_shape[1], input_shape[2]), input_shape=input_shape)(inputs)

sdn = Model(inputs, [warped, disp_M(inputs)])

print(sdn.summary())


disp_M.load_weights(weights_path + weights_name )

print("Loading weights from " + weights_name)

#samples = ['34', '35', '36', '32', '38', '40']
#samples = os.listdir(datapath) + os.listdir('/home/dkuang/LPBA40_npy/image/')

samples = ['{:03d}.npy'.format(i) for i in range(80, 100)] # replace this with your samples' index
sigma = 1

def ini_temp(samples = samples):
    data = 0
    for sample in samples:
      data += np.load(datapath+sample)
    
    data = data/len(samples)
    data = data/np.max(data)
    return data


ini = ini_temp()
#ini = np.load(datapath + samples[0])
#ini = np.load('') 

ini = np.expand_dims(np.expand_dims(ini, 3),0)

#ini = gaussian_filter(ini, sigma)
#ini = np.load('/home/dykuang/MindBoggle/Otemp.npy')
#np.save(weights_path+r'result_test/avg_direct.npy', ini)

#import SimpleITK as sitk
#avg = sitk.ReadImage(r'/home/dkuang/LPBA40/avg152T1_brain.nii')
#avg_data = sitk.GetArrayViewFromImage(avg) 
#ini = avg_data/np.max(avg_data)

'''
test transform
'''
#ini1 = transform(np.zeros([1,res1, res2, res3, 3]), ini, (res1, res2, res3))
#print(np.all(ini1==ini))
#print([np.min(ini1-ini), np.max(ini1-ini)])

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
epsilon = 1e-2
diff = 100
while diff > epsilon:
    warped = np.zeros_like(ini)
    D_sum = len(samples)*np.ones_like(ini)
    D_sum[0,:-1,:-1,:-1,0] = 0
    for sample in samples:
       tgt = np.load(datapath + sample)
       #tgt = gaussian_filter(tgt, sigma) # blurring
       tgt = np.expand_dims(np.expand_dims(tgt, 3),0)
       pair = np.stack([tgt,ini], 4)[...,0] #warp towards current guess 
       
       img, disp = sdn.predict(pair)

       D_phi = Get_Ja(disp)
       warped[0,:-1,:-1,:-1,0] += (D_phi[0])*img[0,:-1,:-1,:-1,0]
       D_sum[0,:-1,:-1,:-1,0] += D_phi[0]
    ini_p = warped/D_sum
    ini_p = ini_p/np.max(ini_p)
    diff = np.mean(np.abs(ini_p-ini))
    print(diff)

    ini = ini_p

print('%s seconds used in average construction.'%(time.time() - start_time))

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
    

'''
Now warp the labels to construct probability map
'''
labelpath = r'path/storing/annotated/labels'
#label_samples = os.listdir(labelpath)
#label_list = np.array([1002, 1003, 1005, 1006, 1007, 1008, 1009, 1011,
#              1012, 1013, 1014, 1015, 1016, 1017, 1018, 1021, 1022,
#              1024, 1025, 1028, 1029, 1030, 1031, 1034, 1035, 2002,
#              2003, 2005, 2006, 2007, 2008, 2009, 2011, 2012, 2013,
#              2014, 2015, 2016, 2017, 2018, 2021, 2022, 2024, 2025,
#              2028, 2029, 2030, 2031, 2034, 2035])

'''
indexes used for labeling
'''
label_list =[0]+[1002, 1003]+[i for i in range(1005, 1032, 1)]+[1034, 1035]+[2002, 2003]+[i for i in range(2005, 2032, 1)] + [2034, 2035]

Vol_Prob = np.zeros([res1,res2,res3,len(label_list)])

from scipy.interpolate import interpn

xx = np.arange(res2)
yy = np.arange(res1)
zz = np.arange(res3)

grid = np.transpose(np.array(np.meshgrid(xx, yy, zz)),(1,2,3,0))
temp = np.load('template/to/warp/to')
#temp = gaussian_filter(temp, 1)
#temp = np.load(datapath + samples[0])
temp = temp/np.max(temp)

temp_label = np.load('labels/with/the/template') # Not necessary, just for comparing purpose
#temp_label = np.load(labelpath+samples[0])
mask = temp>0.01
W = np.ones(len(samples))
#ini = ini[0,...,0]
#temp = ini
for sample_ind, sample in enumerate(samples):
     tgt = np.load(datapath + sample)
     #tgt = np.expand_dims(np.expand_dims(tgt, 3),0)
     pair = np.stack([tgt, temp], 3) #warp towards current guess 
     img, disp = sdn.predict(pair[np.newaxis,])
     #D = np.mean(np.square(disp[0]), axis = None) # a similarity measure
     #W[sample_ind] = np.exp(-D/sigma)
     #DJ = np.exp(-np.square(Get_Ja(disp)[0]-1)/1.0)
     brain_label = np.load(labelpath+sample)
     warped_grid = grid + disp[0]
     warped_grid = np.stack((warped_grid[:, :, :, 1], warped_grid[:, :, :, 0], warped_grid[:, :, :, 2]), 3)

     for label_ind, i in enumerate(label_list):
         Vol_Prob[...,label_ind] += W[sample_ind]*interpn((yy, xx, zz), brain_label==i, warped_grid, method='nearest', bounds_error=False, fill_value=0) # linear interpolation takes more time
         #Vol_Prob[:-1,:-1,:-1,label_ind] += DJ*interpn((yy, xx, zz), brain_label==i, warped_grid, method='nearest', bounds_error=False, fill_value=0)[:-1,:-1,:-1]


#Vol_Prob = np.exp(np.mean(label_prob, axis=0)) #other than taking mean?
#Vol_sum = np.sum(Vol_Prob, axis = 3)

#for i in range(len(label_list)):
#    Vol_Prob[...,i] = Vol_Prob[...,i]/D_sum[0,...,0]
'''
Probability across samples
'''
maxprob = np.amax(Vol_Prob, axis=3)/np.sum(W) 
#np.save(weights_path+'lpba_maxprob_sample.npy', maxprob)
WinningLabel = np.argmax(Vol_Prob,axis=3)
#WinningLabel = WinningLabel*(maxprob>0.02)
'''
Weighted? Joint Fusion?
'''


print('%s seconds used in whole atlas construction.'%(time.time() - start_time))    
np.save(weights_path+'mb_avg_incept0_temp1_near.npy', ini)
#np.save(weights_path+'mb_prob_incept0.npy', Vol_Prob)
np.save(weights_path+'mb_maxprob_temp1_near.npy', maxprob)
np.save(weights_path+ 'mb_WinningLabel_temp1_near.npy', WinningLabel)

dice = [Dice(temp_label==label_list[i], WinningLabel==i) for i in range(1,len(label_list))]

print(np.mean(dice[1:]))
#print(W)

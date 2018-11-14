'''
Evaluate the trained network
'''
import numpy as np
from keras.models import Model
#from spatial_deformer_net3d import SpatialDeformer3D
from architecture import SDN_incept as SDN
from keras.layers import Input
from Utils import Dice
#import os
#import tensorflow as tf

res1, res2, res3 = 144, 180, 144

input_shape = (res1,res2,res3,2)

inputs = Input(shape = input_shape)

sdn = Model(inputs, SDN(inputs)) #change this when using different models

print(sdn.summary())

#sdn.compile(loss = ['mse', total_variation_loss],
#            loss_weights = [1.0, 1e-6],
#            optimizer = 'adam' )

#datapath = r'/home/dkuang/LPBA40/delineation_img_resize/'
#labelpath = r'/home/dkuang/LPBA40/delineation_label_resize/'

datapath = r''
labelpath = r''
sdn.load_weights(r'')

print("Loading weights from ...")


test_files = ['{:03d}'.format(i) for i in range(80, 100)]

label_list_31 = [1002, 1003]+[i for i in range(1005, 1036, 1)]+[2002, 2003]+[i for i in range(2005, 2032, 1)] + [2034, 2035]
label_list = [1002, 1003, 1005, 1006, 1007, 1008, 1009, 1011,
              1012, 1013, 1014, 1015, 1016, 1017, 1018, 1021, 1022,
              1024, 1025, 1028, 1029, 1030, 1031, 1034, 1035, 2002,
              2003, 2005, 2006, 2007, 2008, 2009, 2011, 2012, 2013,
              2014, 2015, 2016, 2017, 2018, 2021, 2022, 2024, 2025,
              2028, 2029, 2030, 2031, 2034, 2035]
test_list = []

from itertools import combinations
for ind in combinations(range(0,len(test_files),1),2):
      test_list.append(test_files[ind[0]]+test_files[ind[1]])
      test_list.append(test_files[ind[1]]+test_files[ind[0]])

print(test_list)

from scipy.interpolate import interpn

xx = np.arange(res2)
yy = np.arange(res1)
zz = np.arange(res3)

grid = np.transpose(np.array(np.meshgrid(xx, yy, zz)),(1,2,3,0))
#grid = np.transpose(grid, (1, 2, 3, 0))
#sample = np.zeros([res1,res2,res3, 3])
#dice_before = np.zeros([len(test_list), 64])

dice_after = np.zeros([len(test_list), 50])
for test_ind, test_label in enumerate(test_list):
    brain_test_1 = np.load(datapath+test_label[:3]+'.npy')
    brain_test_2 = np.load(datapath+test_label[3:6]+'.npy')
    brain_test = np.stack([brain_test_1, brain_test_2], 3)

    brain_test_label1 = np.load(labelpath+test_label[:3]+'.npy')
    brain_test_label2 = np.load(labelpath+test_label[3:6]+'.npy')

    brain_test = np.expand_dims(brain_test, 0)
      
    deformation = sdn.predict(brain_test)[0,...]
    
    sample = grid + deformation 
    
    sample = np.stack((sample[:, :, :, 1], sample[:, :, :, 0], sample[:, :, :, 2]), 3)
    warp_seg = interpn((yy, xx, zz), brain_test_label1, sample, method='nearest', bounds_error=False, fill_value=0)
    
    dice_after[test_ind, :] = np.array([Dice(brain_test_label2==i, warp_seg==i) for i in label_list])
    print('Test sample {}\'s evaluation completed.'.format(test_ind+1))

#    for label_ind, i in enumerate(label_list):
        #dice_before[test_ind, label_ind]= Dice(brain_test_label2==i, brain_test_label1 == i)
#        seg = transform(deformation, np.expand_dims(np.expand_dims(brain_test_label1==i, 3),0), (res1, res2, res3))

#        dice_after[test_ind, label_ind] = Dice(brain_test_label2 == i, np.rint(seg)>0)
    #print("ROI {} result appended..".format(i))
   # print('Test sample {}\'s evaluation completed.'.format(test_ind+1))
#print("writing the first to log....")

np.save('.npy', dice_after)
#np.save('/global/home/hpc4355/MindBoggle/output/warped.npy', warp_seg)


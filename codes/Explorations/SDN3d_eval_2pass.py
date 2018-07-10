'''
Evaluate the trained network
'''
import numpy as np
from keras.models import Model
#from spatial_deformer_net3d import SpatialDeformer3D
from architecture import SDN_ver1
from keras.layers import Input
from Utils import Dice, transform
import os


res1, res2, res3 = 91, 109, 91
input_shape = (res1,res2,res3,2)

inputs = Input(shape = input_shape)

ctx = Model(inputs, SDN_ver1(inputs)) #change this when using different models
rst = Model(inputs, SDN_ver1(inputs))

datapath = r'/home/dkuang/LPBA40_flirt_2mask/image_test_w_2mask/'
labelpath = r'/home/dkuang/LPBA40_flirt_2mask/label/'

ctx.load_weights(r'result_test/SDN3d_weights_2pass_ctx.h5')
rst.load_weights(r'result_test/SDN3d_weights_2pass_rst.h5')
print("Loading weights...")
label_test = os.listdir(datapath)

print(label_test)
label_list = [i for i in range(21,35,1)] + [i for i in range(41,51,1)] + [i for i in range(61, 69, 1)] + [i for i in range(81,93,1)] + [101, 102, 121, 122] + [i for i in range(161, 167, 1)] + [181, 182]

dice_before = np.zeros([len(label_test), 56])
dice_after = np.zeros([len(label_test), 56])
for test_ind, test_label in enumerate(label_test):
    brain_test = np.load(datapath+test_label)
    brain_test = np.expand_dims(brain_test, 0)
    brain_test_ctx = brain_test[...,:2]*brain_test[...,2:]
    brain_test_rst = brain_test[...,:2]*(1-brain_test[...,:2]) 

    brain_test_label1 = np.load(labelpath+test_label[:2]+'.npy')
    brain_test_label2 = np.load(labelpath+test_label[2:4]+'.npy')

    brain_test = np.expand_dims(brain_test, 0)
    deformation = ctx.predict(brain_test_ctx)+rst.predict(brain_test_rst)
    for label_ind, i in enumerate(label_list):
        dice_before[test_ind, label_ind]= Dice(brain_test_label2==i, brain_test_label1 == i)
        seg = transform(deformation, np.expand_dims(np.expand_dims(brain_test_label1==i, 3),0), (res1, res2, res3))

        dice_after[test_ind, label_ind] = Dice(brain_test_label2 == i, np.rint(seg)>0)
        #print("ROI {} result appended..".format(i))
    print('Test sample {}\'s evaluation completed.'.format(test_ind+1))

np.save('result_test/dice_after_2pass.npy', dice_after)


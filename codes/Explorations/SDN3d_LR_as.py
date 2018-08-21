'''
Evaluate the trained network
'''
import numpy as np
from keras.models import Model
#from spatial_deformer_net3d import SpatialDeformer3D
from architecture import SDN_ver11 as SDN
from keras.layers import Input
from Utils import interpolate, Jac
import os

res1, res2, res3 = 91, 109, 91
input_shape = (res1,res2,res3,2)

inputs = Input(shape = input_shape)

sdn = Model(inputs, SDN(inputs)) #change this when using different models

print(sdn.summary())


datapath = r'/home/dkuang/LPBA40_npy/image_test/'
LRpath = r'/home/dkuang/LPBA40_LR/'

dispath = r'/home/tschmah/warpnn/ureg6/'

sdn.load_weights(r'result_test/SDN3d_weights_243.h5')

print("Loading weights from SDN3d_weights_243.h5")
datapath2 = r'/home/dkuang/LPBA40_npy_flirt/image_test/'
label_test = os.listdir(datapath2)

print(label_test)
#from Utils1 import interpolate

test_log = open('result_test/SDN3dLRnorm_243.txt', 'w')
test_log.write("Brain pair \t ||u||^2 \t ||u_a||^2\n ")

#label_test = ['3132']
import SimpleITK as sitk
for test_ind, test_label in enumerate(label_test):
    
    brain_test_1 = np.load(datapath+test_label[:2]+'.npy')
    brain_test_2 = np.load(datapath+test_label[2:4]+'.npy')
    brain_test = np.stack([brain_test_1, brain_test_2], 3)

    brain_test = np.expand_dims(brain_test, 0)
    deformation = sdn.predict(brain_test)
    
    flip = np.load(LRpath + test_label[:2] +'LR.npy')
    X = flip[...,0].flatten() 
    Y = flip[...,1].flatten()
    Z = flip[...,2].flatten()
   
#    Xnii = sitk.ReadImage(dispath+'S'+test_label[:2]+'to' +'S'+ test_label[2:4]+'.fsl152.skullstripped_DispField_Trg2Tpl_X.nii')
#    Ynii = sitk.ReadImage(dispath+'S'+test_label[:2]+'to' +'S'+ test_label[2:4]+'.fsl152.skullstripped_DispField_Trg2Tpl_Y.nii')
#    Znii = sitk.ReadImage(dispath+'S'+test_label[:2]+'to' +'S'+ test_label[2:4]+'.fsl152.skullstripped_DispField_Trg2Tpl_Z.nii')

#    dispX = sitk.GetArrayViewFromImage(Xnii)
#    dispY = sitk.GetArrayViewFromImage(Ynii)
#    dispZ = sitk.GetArrayViewFromImage(Znii)
#    deformation = np.stack([dispX, dispY, dispZ], 3)
#    deformation = np.expand_dims(deformation, 0)
    
    disp_LR = interpolate(deformation, Y, Z, X, (91,109,91))
    disp_LR = np.reshape(disp_LR, (91,109,91,3))
        
    disp_pf = np.einsum('abcde,abce->abcd', Jac(flip), disp_LR[:-1,:-1,:-1,:])
    disp_N = np.mean(deformation**2)     
    disp_LR_N = 0.25*np.mean((deformation[0,:-1,:-1,:-1,:]+disp_pf)**2)
    test_log.write('{}\t'.format(test_label))
    test_log.write('{:6f}\t'.format(disp_N))
    test_log.write('{:6f}\n'.format(disp_LR_N))
    print('Test sample {}\'s evaluation completed.'.format(test_ind+1))
    #print([disp_N, disp_LR_N])
test_log.close()



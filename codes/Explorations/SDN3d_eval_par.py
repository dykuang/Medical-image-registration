'''
Compare with different registration tools
'''
import numpy as np
from keras.models import Model
from architecture import SDN_ver1
from keras.layers import Input
from Utils import Dice, transform
import SimpleITK as sitk

res1, res2, res3 = 91, 109, 91
input_shape = (res1,res2,res3,2)

inputs = Input(shape = input_shape)

sdn = Model(inputs, SDN_ver1(inputs))
sdn.load_weights(r'/home/dkuang/Github/Medical-image-registration/result_test/SDN3d_weights_v1.h5')

testlist = open("result_test/testlist.txt", 'r')
label_test = []
for line in testlist:
    label_test.append(line[:-1])
testlist.close()

label_test=label_test[:25] 
print(label_test)
datapath = r'/home/dkuang/LPBA40_npy_flirt/image/'
labelpath = r'/home/dkuang/LPBA40_npy_flirt/label/'

utilimgpath = r'/home/tschmah/warpnn/ureg/'
utillabelpath = r'/home/tschmah/warpnn/ureg/'


label_list = [i for i in range(21,35,1)] + [i for i in range(41,51,1)] + [i for i in range(61, 69, 1)] + [i for i in range(81,93,1)] + [101, 102, 121, 122] + [i for i in range(161, 167, 1)] + [181, 182]

dice_before = np.zeros([len(label_test), 56])
dice_after_sdn = np.zeros([len(label_test), 56])
dice_after_utr = np.zeros([len(label_test), 56])

sse_before = np.zeros(len(label_test))
sse_sdn = np.zeros(len(label_test))
sse_utr = np.zeros(len(label_test))
for ind, test_label in enumerate(label_test):
    mov = test_label[:2]
    fix = test_label[2:4]
    brain_utr = sitk.ReadImage(utilimgpath+'S{}toS{}.fsl152.skullstripped_FinalDefSrc.nii'.format(mov, fix))
    brain_utr_data = sitk.GetArrayViewFromImage(brain_utr)
    utr_warped_img = brain_utr_data/np.max(brain_utr_data)

    label_utr = sitk.ReadImage(utillabelpath + 'S{}toS{}.fsl152.structure.label.nii'.format(mov, fix))
    utr_warped_label = sitk.GetArrayViewFromImage(label_utr)

    brain_test = np.load(datapath + test_label)
    brain_test_label1 = np.load(labelpath+'{}.npy'.format(mov))
    brain_test_label2 = np.load(labelpath+'{}.npy'.format(fix))

    brain_test = np.expand_dims(brain_test, 0)
    deformation = sdn.predict(brain_test)

    sdn_warped_img = transform(deformation, np.expand_dims(brain_test[...,0],4), (res1, res2, res3))


    sse_before[ind] = np.sum((brain_test[...,0] - brain_test[...,1])**2)
    sse_sdn[ind] = np.sum((brain_test[0,...,1]-sdn_warped_img[0,...,0])**2)
    sse_utr[ind] = np.sum((brain_test[0,...,1]-utr_warped_img)**2)

#print("sse_before: {}".format(sse_before))
#print("sse_sdn: {}".format(sse_sdn))
#print("sse_utr: {}".format(sse_utr))

    for label_ind, i in enumerate(label_list):
        dice_before[ind, label_ind]= Dice(brain_test_label2==i, brain_test_label1 == i)
        seg = transform(deformation, np.expand_dims(np.expand_dims(brain_test_label1==i, 3),0), (res1, res2, res3))
        #seg_utr = transform(disp, np.expand_dims(np.expand_dims(brain_test_label1==i, 3),0), (res1, res2, res3))
        dice_after_sdn[ind,label_ind] = Dice(brain_test_label2 == i, np.rint(seg)>0)
        dice_after_utr[ind,label_ind] = Dice(brain_test_label2 == i, np.rint(utr_warped_label)==i)
    print("{}-th sample done.".format(ind+1))
        
#print("utr vs before: {} ROIs are better".format(np.sum(dice_after_utr[0]>dice_before[0])))
#print("sdn vs before: {} ROIs are better".format(np.sum(dice_after_sdn[0]>dice_before[0])))
#print("utr vs sdn: {} ROIs are better".format(np.sum(dice_after_utr[0]>dice_after_sdn[0])))

#sitk.WriteImage(sdn_warped_img, r'result_test/sdn12to16.nii')
np.save(r'/home/dkuang/Github/Medical-image-registration/result_test/dice_BeforeReg.npy', dice_before)
np.save(r'/home/dkuang/Github/Medical-image-registration/result_test/dice_after_sdn.npy', dice_after_sdn)
np.save(r'/home/dkuang/Github/Medical-image-registration/result_test/dice_after_utr.npy', dice_after_utr)
#np.save(r'/home/dkuang/Github/Medical-image-registration/result_test/sdn12to16.npy', sdn_warped_img)

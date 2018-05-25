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

exp_ind = 6 

res1, res2, res3 = 91, 109, 91
input_shape = (res1,res2,res3,2)

inputs = Input(shape = input_shape)

sdn = Model(inputs, SDN_ver1(inputs)) #change this when using different models

#sdn.compile(loss = ['mse', total_variation_loss],
#            loss_weights = [1.0, 1e-6],
#            optimizer = 'adam' )

datapath = r'/home/dkuang/LPBA40_npy_flirt/image_test/'
labelpath = r'/home/dkuang/LPBA40_npy_flirt/label/'

sdn.load_weights(r'result_test/SDN3d_weights_whole.h5')

#testlist = open("result_test/testlist.txt", 'r')
#label_test = []
#for line in testlist:
#    label_test.append(line[:-1])
#testlist.close()

print("Loading weights from SDN3d_weights_whole.h5")
label_test = os.listdir(datapath)

print(label_test)
label_list = [i for i in range(21,35,1)] + [i for i in range(41,51,1)] + [i for i in range(61, 69, 1)] + [i for i in range(81,93,1)] + [101, 102, 121, 122] + [i for i in range(161, 167, 1)] + [181, 182]

dice_before = np.zeros([len(label_test), 56])
dice_after = np.zeros([len(label_test), 56])
for test_ind, test_label in enumerate(label_test):
    brain_test = np.load(datapath+test_label)
    brain_test_label1 = np.load(labelpath+test_label[:2]+'.npy')
    brain_test_label2 = np.load(labelpath+test_label[2:4]+'.npy')

    brain_test = np.expand_dims(brain_test, 0)
    deformation = sdn.predict(brain_test)
    for label_ind, i in enumerate(label_list):
        dice_before[test_ind, label_ind]= Dice(brain_test_label2==i, brain_test_label1 == i)
        seg = transform(deformation, np.expand_dims(np.expand_dims(brain_test_label1==i, 3),0), (res1, res2, res3))

        dice_after[test_ind, label_ind] = Dice(brain_test_label2 == i, np.rint(seg)>0)
        #print("ROI {} result appended..".format(i))
    print('Test sample {}\'s evaluation completed.'.format(test_ind+1))
#print("writing the first to log....")
#test_log = open('result_test/SDN3dTest.txt', 'w')
#test_log.write("Brain pair: {}\n".format(label_test[0]))
#test_log.write("*"*40+"\n")
#test_log.write("Before Registration"+"\t" + "After Registration\n")
#for before, after in zip(dice_before[0], dice_after[0]):
    #test_log.write("{0:.6f}".format(before)+"\t" + "{0:.6f}".format(after)+'\n')

#test_log.close()

#np.save('result_test/dice_before_1_2.npy', dice_before)
np.save('result_test/dice_after_whole.npy', dice_after)


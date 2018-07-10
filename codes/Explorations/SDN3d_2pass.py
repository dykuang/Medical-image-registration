'''
This script tries the idea of using two different passes that
trains displacement field on cortex and non-cortex brain part separately
'''

import numpy as np
from architecture import SDN_ver1
from keras.models import Model
from keras.optimizers import Adam
import keras.backend as K
from keras.layers import Lambda, add, Input
from spatial_deformer_net3d import SpatialDeformer3D

par = {'res1': 91,

       'res2': 109,

       'res3': 91,

       'kernel_size': (2,2,2),

       'kernel_strides': 2,

       'loss_weights': [1, 2e-5, 2e-5],

       'epochs': 10,

       'batch_size': 5,

       'lr': 1e-3,

       'w1': 3,

       'w2': 1,

       'w1_rst': 3,

       'w2_rst': 1,
       }

print(par)

res1 = par['res1']

res2 = par['res2']

res3 = par['res3']


def split_by_mask(args):
    ipts, mask = args[...,:2], args[...,2:]
    return [ipts*mask, ipts*(1-mask)]

'''
Modify this and the extract file to provide x|y|x_mask|y_mask|
'''
def vol_generator(path, file_list, batch_size):
    x = np.zeros((batch_size, res1, res2, res3, 4))
    zeros = np.zeros([batch_size, res1, res2, res3, 3])
    count = 0
    while True:
        for j in range(batch_size):
            x[j] = np.load(path+file_list[(count*batch_size+j)%len(file_list)])

        yield split_by_mask(x), [np.expand_dims(x[...,1],4), zeros, zeros]

        count = count + 1
        if count >= len(file_list)/batch_size:
            count = 0

def vol_generator2(path, label_path, file_list, batch_size):
    x = np.zeros((batch_size, res1, res2, res3, 1))
    x_ctx = np.zeros((batch_size, res1, res2, res3, 2))
    x_rst = np.zeros((batch_size, res1, res2, res3, 2))
    zeros = np.zeros([batch_size, res1, res2, res3, 3])
    count = 0
    while True:
        for j in range(batch_size):
            pair = file_list[(count*batch_size+j)%len(file_list)]
            mov = np.load(path+pair[:2]+'.npy')
            fix = np.load(path+pair[2:]+'.npy')
            mov_mask = np.load(label_path+pair[:2]+'.npy')
            fix_mask = np.load(label_path+pair[2:]+'.npy')
            mov_mask_ctx = (mov_mask > 0) & (mov_mask<130)
            fix_mask_ctx = (fix_mask > 0) & (fix_mask<130)
            mov_mask_rst = mov_mask > 130
            fix_mask_rst = fix_mask > 130
            
            x_ctx[j] = np.stack([mov*mov_mask_ctx, fix*fix_mask_ctx], 3)
            x_rst[j] = np.stack([mov*mov_mask_rst, fix*fix_mask_rst], 3)
            x[j,...,0] = fix

        yield [x_ctx, x_rst], [x, zeros, zeros]

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
    diff = yPred  # yTrue = 0, ignored.

    return par['w1']*total_variation(diff) + par['w2']*K.pow(K.sum(K.pow(diff, 2)),0.5)

def total_variation_loss_rst(yTrue, yPred):
#    assert K.ndim(yTrue) == 4
    diff = yPred  # yTrue = 0, ignored.

    return par['w1_rst']*total_variation(diff) + par['w2_rst']*K.pow(K.sum(K.pow(diff, 2)),0.5)

def get_masked(args):
    ipts, mask = args[...,:2], args[...,2:]
    print('shape of maksed:{}'.format(K.int_shape(mask)))
    print('shape of multiplied:{}'.format(K.int_shape(ipts*(mask))))
    return ipts*mask

def get_unmasked(args):
    ipts, mask = args[...,:2], args[...,2:]
    print('shape of unmaksed:{}'.format(K.int_shape(1-mask)))
    print('shape of multiplied:{}'.format(K.int_shape(ipts*(1-mask))))    
    return ipts*(1-mask)

'''
Commented out data can all work..
need a different dataset..
'''

input_shape = (res1,res2,res3,2)
#inputs = Input(shape = input_shape)
#print(K.int_shape(inputs))
#splited = Lambda(split_by_mask)(inputs) #considerusing tf.split?
#ctx, rst = splited[0], splited[1]
#ctx = Lambda(get_masked)(inputs)
#rst = Lambda(get_unmasked)(inputs)
#print(K.int_shape(ctx))
#print(K.int_shape(rst))

ctx = Input((res1, res2, res3, 2))
rst = Input((res1, res2, res3, 2))

disp_ctx = SDN_ver1(ctx)
disp_ctx_M = Model(ctx, disp_ctx)
print(disp_ctx_M.summary())
img_ctx = SpatialDeformer3D(localization_net=disp_ctx_M,
                             output_size=(input_shape[0],input_shape[1], input_shape[2]),
                             input_shape=input_shape)(ctx)
disp_rst = SDN_ver1(rst)
disp_rst_M = Model(rst, disp_rst)
print(disp_rst_M.summary())
img_rst = SpatialDeformer3D(localization_net=disp_rst_M,
                             output_size=(input_shape[0],input_shape[1], input_shape[2]),
                             input_shape=input_shape)(rst)

disp = add([disp_ctx, disp_rst])
img = add([img_ctx, img_rst])

print(K.int_shape(disp))
print(K.int_shape(img))

sdn = Model([ctx, rst], [img,disp_ctx, disp_rst])
#disp_whole = Model([ctx, rst], disp)
#print(disp_whole.summary())
#warped = SpatialDeformer3D(localization_net=disp_whole,
#                             output_size=(input_shape[0],input_shape[1], input_shape[2]),
#                             input_shape=input_shape)(inputs)

#sdn = Model([inputs, ctx, rst], [warped, disp])
#sdn1 = Model(inputs, [warped, disp_ctx(ctx), disp_rst(rst)])
print(sdn.summary())
#print(sdn1.summary())
sdn.compile(loss = ['mse', total_variation_loss, total_variation_loss_rst],
            loss_weights = par['loss_weights'],
            optimizer = Adam(lr = par['lr'], decay = 1e-5) )


epochs = par['epochs']
batch_size = par['batch_size']

import os
datapath = r'/LPBA40_npy_flirt/image/'
labelpath = r'/LPBA40_npy_flirt/label/'
testpath = r'/LPBA40_npy_flirt/image_test/'
datalist = os.listdir(datapath)
testlist = os.listdir(testpath)
#print(datalist)
#random.shuffle(datalist)
#train_list = datalist[:]
#test_list = testlist[:]
#train_list = ['0520.npy']
#test_list = ['0520.npy']
#print(train_list)
#print(len(train_list))
#validation_list = testlist[:]

train_list = []
validation_list=[]

from itertools import combinations
for ind in combinations(range(1,31,1),2):
    train_list.append('{0:02d}{1:02d}'.format(ind[0],ind[1]))
    train_list.append('{0:02d}{1:02d}'.format(ind[1],ind[0]))
for ind in combinations(range(31,41,1),2):
    validation_list.append('{0:02d}{1:02d}'.format(ind[0],ind[1]))
    validation_list.append('{0:02d}{1:02d}'.format(ind[1],ind[0]))


gen_train = vol_generator2(datapath, labelpath, train_list, batch_size)
gen_val = vol_generator2(testpath, labelpath, validation_list, batch_size)
history = sdn.fit_generator(gen_train, steps_per_epoch = len(train_list)/batch_size, epochs = epochs, use_multiprocessing = True, verbose=1, validation_data = gen_val, validation_steps = len(validation_list)/batch_size)
loss = history.history['loss']
val_loss = history.history['val_loss']
print("Training complete.")
disp_ctx_M.save_weights('/output/SDN3d_weights_2pass_ctx.h5')
disp_rst_M.save_weights('/output/SDN3d_weights_2pass_rst.h5')

print("Saving current model ...")

 
from sklearn.metrics import jaccard_similarity_score
def j_score(yTrue, yPred):
     js=[]
     for yT, yP in zip(yTrue, yPred):
          js.append(jaccard_similarity_score((yT>0.1).flatten(), (yP>0.1).flatten()))
     js = np.stack(js)
     return np.mean(js)

# test on one brain
label = '3132.npy'
#brain_test1_mov = np.load(testpath+label[:2] + '.npy')
#brain_test1_fix = np.load(testpath+label[2:] + '.npy')

#brain_test1_mov_mask = np.load(maskpath+label[:2] + '.npy')


#brain_test_label1 = np.load(labelpath+label[:2]+'.npy')
#brain_test_label2 = np.load(labelpath+label[2:4]+'.npy')

def Dice(y_true, y_pred):
     T = (y_true.flatten()>0)
     P = (y_pred.flatten()>0)

     return 2*np.sum(T*P)/(np.sum(T) + np.sum(P))


#label1_data = np.rint(brain_test_label1)
#label2_data = np.rint(brain_test_label2)

#label_list = list(np.unique(label1_data))[1:]
label_list = [i for i in range(21,35,1)] + [i for i in range(41,51,1)] + [i for i in range(61, 69, 1)] + [i for i in range(81,93,1)] + [101, 102, 121, 122] + [i for i in range(161, 167, 1)] + [181, 182]


dice_before, dice_after = [], []

'''
Different ways to warp the label mask makes huge differences...
'''
#brain_test1 = np.expand_dims(brain_test1, 0)
#warped_brain, deformation = sdn.predict(brain_test1)

gen_test = vol_generator2(testpath, labelpath, label, 1)
[warped, d_ctx, d_rst] = sdn.predict_generator(gen_test)
deformation = d_ctx + d_rst

from Utils import transform
for i in label_list:

     dice_before.append(Dice(label2_data==i, label1_data == i))
     seg = transform(deformation, np.expand_dims(np.expand_dims(label1_data==i, 3),0), (res1, res2, res3))

     dice_after.append(Dice(label2_data == i, np.rint(seg)>0))
     #print("ROI {} result appended..".format(i))
count_worse=0
count_equal = 0
count_better = 0

for i in range(56):

    if dice_after[i] < dice_before[i]:

        count_worse += 1

    elif dice_after[i] > dice_before[i]:

        count_better += 1

    else:

        count_equal += 1

print('worse: {}'.format(count_worse))
print('equal: {}'.format(count_equal))
print('better: {}'.format(count_better))

print("writing to log....")
test_log = open('/home/dkuang/SDN3dTest_duopass.txt', 'w')
test_log.write("paramters:\n")
for key, value in par.items():
    test_log.write("{}:\t{}\n".format(key, value))
test_log.write("*"*40+"\n")

test_log.write("training loss:\n")
for item in loss:
    test_log.write("%s\n"% item)
test_log.write("validation loss:\n")
for item in val_loss:
    test_log.write("%s\n"% item)

test_log.write("Brain pair: {}\n".format(label))
test_log.write("Before registration: {}\n".format(j_score(brain_test1[...,1], brain_test1[...,0])))
test_log.write("After registration: {}\n".format(j_score(brain_test1[...,1], warped_brain)))
test_log.write("Before registration: {}\n".format(Dice(brain_test1[...,1], brain_test1[...,0])))
test_log.write("After registration: {}\n".format(Dice(brain_test1[...,1], warped_brain)))
test_log.write("*"*40+"\n")
test_log.write("Before Registration"+"\t" + "After Registration\n")
for before, after in zip(dice_before, dice_after):
    test_log.write("{0:.6f}".format(before)+"\t" + "{0:.6f}".format(after)+'\n')

test_log.close()

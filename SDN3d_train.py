# -*- coding: utf-8 -*-
"""
The script trains SDN3d
"""
import numpy as np
import random
res1, res2, res3 = 91, 109, 91
# Write a generator

def vol_generator(path, file_list, batch_size):
    x = np.zeros((batch_size, res1, res2, res3, 2))
    zeros = np.zeros([batch_size, res1, res2, res3, 3])
    count = 0
    while True:
#        for i in range(int(len(file_list)/batch_size)):
        for j in range(batch_size):
            x[j] = np.load(path+file_list[(count*batch_size+j)%len(file_list)])
                #x[j] = x[j]*(x[j]>0.1)
        yield x, [np.expand_dims(x[...,1],4), zeros]

        count = count + 1
        if count >= len(file_list)/batch_size:
            count = 0

#def get_batch(datapath, file_list, start_ind, batch_size):
#    x = np.zeros((batch_size, res1, res2, res3, 2))
#    zeros = np.zeros([batch_size, res1, res2, res3, 3])
#    for j in range(batch_size):
#        temp = np.load(datapath+file_list[int(start_ind*batch_size+j)%len(file_list)])
        #print(temp.shape)
#        x[j] = temp
#    return  x, [np.expand_dims(x[...,1],4), zeros]



# Training
#from spatial_deformer_net3d import SpatialDeformer3D
from keras.models import Model
from keras.layers import Input
from keras.optimizers import Adam
from keras.losses import binary_crossentropy, kullback_leibler_divergence
import keras.backend as K
#from keras.regularizers import l1, l2

  
def total_variation(y):
#    assert K.ndim(y) == 4
    a = K.square(y[:, :res1 - 1, :res2 - 1, :res3-1, :] - y[:, 1:, :res2 - 1, :res3-1, :])
    b = K.square(y[:, :res1 - 1, :res2 - 1, :res3-1, :] - y[:, :res1 - 1, 1:, :res3-1, :])
    c = K.square(y[:, :res1 - 1, :res2 - 1, :res3-1, :] - y[:, :res1 - 1, :res2-1, 1:, :])
    
    return K.pow(K.sum(a + b + c), 0.5)# tweak the power?

def total_variation_loss(yTrue, yPred):
#    assert K.ndim(yTrue) == 4
    diff = yTrue - yPred

    return 2*total_variation(diff) + 1*K.pow(K.sum(K.pow(diff, 2)),0.5)


def customloss(yTrue, yPred):
     sse = K.sum(K.square(yTrue - yPred))
     
     Dx_yTrue = yTrue[:, :res1 - 1, :res2 - 1, :res3-1, :] - yTrue[:, 1:, :res2 - 1, :res3-1, :]
     Dy_yTrue = yTrue[:, :res1 - 1, :res2 - 1, :res3-1, :] - yTrue[:, :res1 - 1, 1:, :res3-1, :]
     Dz_yTrue = yTrue[:, :res1 - 1, :res2 - 1, :res3-1, :] - yTrue[:, :res1 - 1, :res2-1, 1:, :]
     
     Dx_yPred = yPred[:, :res1 - 1, :res2 - 1, :res3-1, :] - yPred[:, 1:, :res2 - 1, :res3-1, :]
     Dy_yPred = yPred[:, :res1 - 1, :res2 - 1, :res3-1, :] - yPred[:, :res1 - 1, 1:, :res3-1, :]
     Dz_yPred = yPred[:, :res1 - 1, :res2 - 1, :res3-1, :] - yPred[:, :res1 - 1, :res2-1, 1:, :]
     
     D1 = K.sum(K.square(Dx_yTrue - Dx_yPred))
     D2 = K.sum(K.square(Dy_yTrue - Dy_yPred))
     D3 = K.sum(K.square(Dz_yTrue - Dz_yPred))
     
     return sse+ 0.2*(D1+D2+D3)

from spatial_deformer_net3d import SpatialDeformer3D
from architecture import SDN_ver3

input_shape = (res1,res2,res3,2)
inputs = Input(shape = input_shape)
locnet = Model(inputs, SDN_ver3(inputs))
x1 = SpatialDeformer3D(localization_net=locnet,
                       output_size=(input_shape[0],input_shape[1], input_shape[2]),
                       input_shape=input_shape)(inputs)
      
sdn = Model(inputs, [x1,SDN_ver3(inputs)])

#print(sdn.summary())
print(locnet.summary())
#print(sdn.layers)
from keras.optimizers import SGD
sdn.compile(loss = ['mse', total_variation_loss],
            loss_weights = [1.0, 0],
            optimizer = Adam(lr = 5e-4, decay = 1e-5) )


epochs = 20 
batch_size = 8 
import os
datapath = r'/LPBA40_npy_flirt/image/'
labelpath = r'/LPBA40_npy_flirt/label/'
train_list = os.listdir(datapath)
#random.shuffle(train_list)
train_list = train_list[:]

testpath = r'/LPBA40_npy_flirt/image_test/'
testlist = os.listdir(testpath)
validation_list = testlist[:]

#trainlist = open("result_test/trainlist.txt", 'r')
#train_list = []
#for line in trainlist:
    #train_list.append(line[:-1])
#trainlist.close()

#random.seed(1234)
#random.shuffle(datalist)
#train_list = datalist[:280]
#test_list = datalist[-100:]


#trainlist = open('result_test/trainlist.txt', 'w')
#for f in train_list:
    #trainlist.write(f)
    #trainlist.write("\n")
#trainlist.close()

#testlist = open('result_test/testlist.txt', 'w')
#for f in test_list:
    #testlist.write(f)
    #testlist.write("\n")
#testlist.close()

"""
Train with fit_generator
"""
gen_train = vol_generator(datapath, train_list, batch_size)
gen_test = vol_generator(testpath, validation_list, batch_size)
history = sdn.fit_generator(gen_train, steps_per_epoch = len(train_list)/batch_size, epochs = epochs, use_multiprocessing = True, verbose=1, validation_data = gen_test, validation_steps = len(validation_list)/batch_size)
loss = history.history['loss']
val_loss = history.history['val_loss']


"""
Now try using train_on_batch
"""
#loss=[]
#for i in range(epochs):
#    print("{0}/{1} epoch:\n".format(i+1, epochs))

#    for j in range(int(np.floor(len(train_list)/batch_size))):
#        X, Y = get_batch(datapath, train_list, j, batch_size)
#        history = sdn.train_on_batch(X, Y)
#        loss.append(history)
#        print("{}-th, minibatch done.\n".format(j+1))
    
#    random.shuffle(train_list)

print("Training complete.")
print("Saving current model ...")
locnet.save_weights('/output/SDN3d_weights_ver5.h5')  # save weights instead?

loss_history = open('/output/loss_history_ver5.txt', 'w')
loss_history.write("training loss:\n")
for item in loss:
    loss_history.write("%s\n"% item)
loss_history.write("validation loss:\n")
for item in val_loss:
    loss_history.write("%s\n"% item)
loss_history.close()
print("Saving complete!")


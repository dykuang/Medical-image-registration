# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 16:10:32 2018

@author: Dongyang


Test the SDN module
"""

import numpy as np
np.random.seed(1337)  # for reproducibility
import matplotlib.pyplot as plt
from scipy.misc import imresize
from keras.datasets import mnist
from keras.models import Model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, UpSampling2D, BatchNormalization
from keras.utils import np_utils
from keras.utils import np_utils, generic_utils
from keras.optimizers import Adam, SGD

import keras.backend as K
from spatial_deformer_net import SpatialDeformer
from spatial_transformer_net import SpatialTransformer

#batch_size = 128
nb_classes = 10
#nb_epoch = 12

DIM = 60
mnist_cluttered = "datasets/mnist_cluttered_60x60_6distortions.npz"

data = np.load(mnist_cluttered)
X_train, y_train = data['x_train'], np.argmax(data['y_train'], axis=-1)
X_valid, y_valid = data['x_valid'], np.argmax(data['y_valid'], axis=-1)
X_test, y_test = data['x_test'], np.argmax(data['y_test'], axis=-1)
# reshape for convolutions
X_train = X_train.reshape((X_train.shape[0], DIM, DIM, 1))
X_valid = X_valid.reshape((X_valid.shape[0], DIM, DIM, 1))
X_test = X_test.reshape((X_test.shape[0], DIM, DIM, 1))

y_train = np_utils.to_categorical(y_train, nb_classes)
y_valid = np_utils.to_categorical(y_valid, nb_classes)
y_test = np_utils.to_categorical(y_test, nb_classes)

print("Train samples: {}".format(X_train.shape))
print("Validation samples: {}".format(X_valid.shape))
print("Test samples: {}".format(X_test.shape))


input_shape =  np.squeeze(X_train.shape[1:])
input_shape = (60,60,1)
print("Input shape:",input_shape)


#locnet = Sequential()
#
#locnet.add(Conv2D(20, (5, 5), padding = 'same', input_shape=input_shape))
#locnet.add(Activation('relu'))
#locnet.add(MaxPooling2D(pool_size=(2,2)))
#locnet.add(Conv2D(20, (5, 5), padding = 'same'))
##locnet.add(BatchNormalization())
#locnet.add(Activation('relu'))

#locnet.add(MaxPooling2D(pool_size=(2,2)))
#locnet.add(Flatten())
#locnet.add(Dense(50))
#locnet.add(Activation('relu'))
#locnet.add(Dense(2, weights=weights)) # what if only do shift?

#locnet.add(UpSampling2D( (2, 2) ))
#locnet.add(Conv2D(20, (5,5), padding = 'same')) # Transpose/Deconvolve or not?
#locnet.add(UpSampling2D( (2, 2)))
#locnet.add(Conv2D(2, (5,5), padding = 'same',
#                  kernel_initializer='zeros',
#                  bias_initializer = 'zeros'))
#locnet.add(Activation('linear'))

b = np.zeros((2, 3), dtype='float32')
b[0, 0] = 1
b[1, 1] = 1
W = np.zeros((50, 6), dtype='float32')
weights = [W, b.flatten()]

#locnet_a = Sequential()
#locnet_a.add(MaxPooling2D(pool_size=(2,2), input_shape=input_shape))
#locnet_a.add(Conv2D(20, (5, 5)))
#locnet_a.add(MaxPooling2D(pool_size=(2,2)))
#locnet_a.add(Conv2D(20, (5, 5)))
#
#locnet_a.add(Flatten())
#locnet_a.add(Dense(50))
#locnet_a.add(Activation('relu'))
#locnet_a.add(Dense(6, weights=weights))



#model = Sequential()
#
#model.add(SpatialTransformer(localization_net=locnet_a,
#                             output_size=(60,60), 
#                             input_shape=input_shape))
#
#model.add(SpatialDeformer(localization_net=locnet,
#                             output_size=(30,30),  # this affects the grid size, should be the same as output of above for deformation
#                             input_shape=input_shape))
#
#model.add(Conv2D(32, (3, 3), padding='same'))
#model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Conv2D(32, (3, 3)))
#model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#
#model.add(Flatten())
#model.add(Dense(256))
#model.add(Activation('relu'))
#
#model.add(Dense(nb_classes))
#model.add(Activation('softmax'))

#model.compile(loss='categorical_crossentropy',
#              optimizer='adam'
#              optimizer=SGD(lr=0.005, decay=1e-6, momentum=0.9, nesterov=True)
#              )
# optimizers.SGD(lr=0.005, decay=1e-6, momentum=0.9, nesterov=True)

inputs = Input(shape = input_shape)

yy = MaxPooling2D((2,2))(inputs)
yy = Conv2D(20, (5,5))(yy)
yy = MaxPooling2D((2,2))(yy)
yy = Conv2D(20, (5,5))(yy)
yy = Flatten()(yy)
yy = Dense(50)(yy)
yy = Dense(6, weights=weights)(yy)
locnet_a = Model(inputs, yy)


zz = Conv2D(20, (5,5), padding = 'same')(inputs)
zz = MaxPooling2D((2,2))(zz)
zz = Conv2D(20, (5,5), padding = 'same')(zz)
#zz = BatchNormalization()(zz)   # causing errors when compiling, need to set scope?
zz = Conv2D(2, (5,5), padding = 'same',
                  kernel_initializer='zeros',
                  bias_initializer = 'zeros',
                  activation = 'tanh')(zz) #careful about the activation
locnet = Model(inputs, zz)

x = SpatialTransformer(localization_net=locnet_a,
                             output_size=(60,60), 
                             input_shape=input_shape)(inputs)

x = SpatialDeformer(localization_net=locnet,
                             output_size=(30,30), 
                             input_shape=input_shape)(x)

x = Conv2D(32, (3, 3), padding='same', activation = 'relu')(x)
x = MaxPooling2D((2,2))(x)
x = Conv2D(32, (3, 3), padding='same', activation = 'relu')(x)
x = MaxPooling2D((2,2))(x)
x = Flatten()(x)
x = Dense(256, activation = 'relu')(x)
x = Dense(nb_classes, activation = 'softmax')(x)
model = Model(inputs, x)


from keras.losses import categorical_crossentropy
def Reg_loss(y_true, y_pred):
    cls_loss = categorical_crossentropy(y_true, y_pred)
    
    return cls_loss 

model.compile(loss = Reg_loss,
              optimizer='adam')

XX = inputs
YY = model.layers[2].output
F = K.function([XX], [YY])

XX_loc = inputs
DD = locnet.output
DF = K.function([XX_loc], [DD])

nb_epochs = 2 # you probably want to go longer than this
batch_size = 128
#fig = plt.figure()
try:
    for e in range(nb_epochs):
        print('-'*40)
        #progbar = generic_utils.Progbar(X_train.shape[0])
        for b in range(150):
            #print(b)
            f = b * batch_size
            l = (b+1) * batch_size
            X_batch = X_train[f:l].astype('float32')
            y_batch = y_train[f:l].astype('float32')
            loss = model.train_on_batch(X_batch, y_batch)
            #print(loss)
            #progbar.add(X_batch.shape[0], values=[("train loss", loss)])
        scorev = model.evaluate(X_valid, y_valid, verbose=0)
        scoret = model.evaluate(X_test, y_test, verbose=0)
        print('Epoch: {0} | Valid: {1} | Test: {2}'.format(e, scorev, scoret))
        
#        if e % 1 == 0:
#            Xresult = F([X_batch[:9]])
#            plt.clf()
#            for i in range(9):
#                plt.subplot(3, 3, i+1)
#                image = np.squeeze(Xresult[0][i])
#                plt.imshow(image, cmap='gray')
#                plt.axis('off')
#            fig.canvas.draw()
#            plt.show()
        
except KeyboardInterrupt:
    pass


Xaug = X_train[:9]
Xresult = F([Xaug.astype('float32')])
xdeform = DF([Xaug]) # the deformation on the reference frame, needed to be rescaled.

plt.figure() 
for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.imshow(np.squeeze(Xaug[i]), cmap='gray')
    plt.axis('off')

plt.figure()    
for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.imshow(np.squeeze(Xresult[0][i]), cmap='gray')
    plt.axis('off')
   
# displacement on the reference frame    
plt.figure()    
for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.imshow(np.squeeze(xdeform[0][i])[:,:,0], cmap='gray')
    plt.axis('off')
    
plt.figure()    
for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.imshow(np.squeeze(xdeform[0][i])[:,:,1], cmap='gray')
    plt.axis('off')
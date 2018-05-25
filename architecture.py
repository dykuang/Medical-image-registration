'''
Architectures for producing the displacement field
'''
#from keras.models import Model
#from spatial_deformer_net3d import SpatialDeformer3D
from keras.layers import Input, Conv3D, MaxPooling3D,  UpSampling3D, multiply, Conv3DTranspose, ZeroPadding3D, Cropping3D, PReLU, concatenate
#import keras.backend as K


par = {
       'kernel_size': (2,2,2),

       'kernel_strides': 2,

      }

def SDN(inputs):

    zz = Conv3D(64, (2,2,2), padding = 'same')(inputs)
    zzz = Conv3D(64, (2,2,2), padding = 'same')(zz)

    zzz = MaxPooling3D((2,2,2))(zzz)
    zzz = Conv3D(128, (2,2,2), padding = 'same')(zzz)

    zzz = UpSampling3D((2,2,2))(zzz)
    zzz = Conv3D(64, (2,2,2), padding = 'same')(zzz)

    zzzz = multiply([zz, zzz])   # easy to cause trouble when shape does not contain enough power of 2.
    zzzz = Conv3D(3, (2,2,2), padding = 'same',
                      kernel_initializer= 'he_normal',
                      bias_initializer = 'he_normal',
#                      activity_regularizer = l2(0.1),
                      activation = 'tanh')(zzz)

#    locnet = Model(inputs, zzzz)

    #x1 = SpatialDeformer3D(localization_net=locnet,
    #                         output_size=(input_shape[0],input_shape[1], input_shape[2]),
    #                         input_shape=input_shape)(inputs)

    return zzzz


def SDN_deeper(inputs): # need a deeper one for 3d? one with no pooling?
    z1_1 = Conv3D(32, (2,2,2), padding = 'same')(inputs)
    z1_2 = Conv3D(32, (2,2,2), padding = 'same')(z1_1)

    z2 = MaxPooling3D((2,2,2))(z1_2)
    z2_1 = Conv3D(64, (2,2,2), padding = 'same')(z2)
    z2_2 = Conv3D(64, (2,2,2), padding = 'same')(z2_1)

    z3 = MaxPooling3D((2,2,2))(z2_2)
    z3 = Conv3D(128, (2,2,2), padding = 'same')(z3)


    z3 = UpSampling3D((2,2,2))(z3)
    z3 = Conv3D(64, (2,2,2), padding = 'same')(z3) # help to overcome local minimum?
#    z3 = Conv3DTranspose(64, (2,2,2), strides=(2,2,2), padding = 'same')(z3)
#    z4 = multiply([z2_1, z3]) 

    z4 = UpSampling3D((2,2,2))(z3)
    z5 = Conv3D(32, (2,2,2), padding = 'same')(z4)
#    z5 = Conv3DTranspose(32, (2,2,2), strides=(2,2,2), padding = 'same')(z3)
#    z5= multiply([z1_2, z4])    


    zzzz = Conv3D(3, (2,2,2), padding = 'same',
#                      kernel_initializer= 'he_uniform',
#                      bias_initializer = 'he_uniform',
#                      activity_regularizer = l1(0.001),
                      activation = 'tanh')(z5)

#    locnet = Model(inputs, zzzz)

    #x1 = SpatialDeformer3D(localization_net=locnet,
    #                         output_size=(input_shape[0],input_shape[1], input_shape[2]),
    #                         input_shape=input_shape)(inputs)

    return zzzz


def SDN_ver1(inputs): #should control the size carefully, larger strides to downsample 
    #z1_1 = Conv3D(32, (2,2,2), padding = 'same')(inputs)
    z1_2 = Conv3D(32, (2,2,2), strides = 2, padding = 'valid')(inputs)
    z1_2 = PReLU(shared_axes = [4])(z1_2)

    #z2_1 = Conv3D(64, (2,2,2), padding = 'same')(z1_2)
    z2_2 = Conv3D(64, (2,2,2), strides = 2, padding = 'valid')(z1_2)
    z2_2 = PReLU(shared_axes = [4])(z2_2)

#    z3 = Conv3D(64, (2,2,2), padding = 'same')(z2_2)


    z4 = Conv3DTranspose(64, (2,2,2), strides=2, padding = 'valid')(z2_2)
    z4 = Conv3D(64, (2,2,2), padding = 'same')(z4)

    z5 = Conv3DTranspose(32, (2,2,2), strides=2, padding = 'valid')(z4)
    z5 = Conv3D(32, (2,2,2), padding = 'same', activation = 'linear')(z5)

    z5 = ZeroPadding3D((2,1,2))(z5)     #Extra padding to make size match
    zzzz = Conv3D(3, (2,2,2), padding = 'valid',
#                      kernel_initializer= 'he_uniform',
#                      bias_initializer = 'he_uniform',
#                      activity_regularizer = l1(0.001),
                      activation = 'tanh')(z5)

#    locnet = Model(inputs, zzzz)

    #x1 = SpatialDeformer3D(localization_net=locnet,
    #                         output_size=(input_shape[0],input_shape[1], input_shape[2]),
    #                         input_shape=input_shape)(inputs)

    return zzzz
    

def SDN_ver2(inputs): 
    '''
    Paralleled decoder for each dimension
    '''
#    z1_1 = Conv3D(32, par['kernel_size'], padding = 'same')(inputs)

    z1_2 = Conv3D(32, par['kernel_size'], strides = par['kernel_strides'], padding = 'valid', activation = 'linear')(inputs)
    #z1_2 = BatchNormalization()(z1_2)
    z1_2 = PReLU(shared_axes = [4])(z1_2)
#    z2_1 = Conv3D(64, par['kernel_size'], padding = 'same')(z1_2)

    z2_2 = Conv3D(64, par['kernel_size'], strides = par['kernel_strides'], padding = 'valid', activation = 'linear')(z1_2)
   # z2_2 = BatchNormalization()(z2_2)  
    z2_2 = PReLU(shared_axes = [4])(z2_2)

    z4x = Conv3DTranspose(64, par['kernel_size'], strides= par['kernel_strides'], padding = 'valid', activation = 'linear')(z2_2)
    z4x = Conv3D(64, (2,2,2), padding = 'same')(z4x)
    z5x = Conv3DTranspose(32, par['kernel_size'], strides= par['kernel_strides'], padding = 'valid', activation = 'linear')(z4x)
    z5x = Conv3D(32, (2,2,2), padding = 'same')(z5x)
    z5x = ZeroPadding3D((2,1,2))(z5x)
    zzzzx = Conv3D(1, par['kernel_size'], padding = 'valid', activation = 'tanh')(z5x)

    z4y = Conv3DTranspose(64, par['kernel_size'], strides= par['kernel_strides'], padding = 'valid',activation = 'linear')(z2_2)
    z4y = Conv3D(64, (2,2,2), padding = 'same')(z4y)
    z5y = Conv3DTranspose(32, par['kernel_size'], strides= par['kernel_strides'], padding = 'valid', activation = 'linear')(z4y)
    z5y = Conv3D(32, (2,2,2), padding = 'same')(z5y)
    z5y = ZeroPadding3D((2,1,2))(z5y)
    zzzzy = Conv3D(1, par['kernel_size'], padding = 'valid', activation = 'tanh')(z5y)

    z4z = Conv3DTranspose(64, par['kernel_size'], strides= par['kernel_strides'], padding = 'valid', activation = 'linear')(z2_2)
    z4z = Conv3D(64, (2,2,2), padding = 'same')(z4z)
    z5z = Conv3DTranspose(32, par['kernel_size'], strides= par['kernel_strides'], padding = 'valid', activation = 'linear')(z4z)
    z5z = Conv3D(32, (2,2,2), padding = 'same')(z5z)
    z5z = ZeroPadding3D((2,1,2))(z5z)
    zzzzz = Conv3D(1, par['kernel_size'], padding = 'valid', activation = 'tanh')(z5z)

    zzzz = concatenate([zzzzx, zzzzy, zzzzz], axis = -1)


    return zzzz




def SDN_ver3(inputs): 
    '''
    Uses three different CONV blocks at last in the decoder part for each dimension
    ''' 
    z1_2 = Conv3D(32, par['kernel_size'], strides = par['kernel_strides'], padding = 'valid', activation = 'linear')(inputs)  
    z1_2 = PReLU(shared_axes = [4])(z1_2)
   

    z2_2 = Conv3D(64, par['kernel_size'], strides = par['kernel_strides'], padding = 'valid', activation = 'linear')(z1_2)
    z2_2 = PReLU(shared_axes = [4])(z2_2)

    z4 = Conv3DTranspose(64, par['kernel_size'], strides= par['kernel_strides'], padding = 'valid', activation = 'linear')(z2_2)
    z4 = Conv3D(64, (2,2,2), padding = 'same')(z4)

    z5x = Conv3DTranspose(32, par['kernel_size'], strides= par['kernel_strides'], padding = 'valid', activation = 'linear')(z4)
    z5x = Conv3D(32, (2,2,2), padding = 'same')(z5x)
    z5x = ZeroPadding3D((2,1,2))(z5x)
    zzzzx = Conv3D(1, par['kernel_size'], padding = 'valid', activation = 'tanh')(z5x)

    z5y = Conv3DTranspose(32, par['kernel_size'], strides= par['kernel_strides'], padding = 'valid', activation = 'linear')(z4)
    z5y = Conv3D(32, (2,2,2), padding = 'same')(z5y)
    z5y = ZeroPadding3D((2,1,2))(z5y)
    zzzzy = Conv3D(1, par['kernel_size'], padding = 'valid', activation = 'tanh')(z5y)

    z5z = Conv3DTranspose(32, par['kernel_size'], strides= par['kernel_strides'], padding = 'valid', activation = 'linear')(z4)
    z5z = Conv3D(32, (2,2,2), padding = 'same')(z5z)
    z5z = ZeroPadding3D((2,1,2))(z5z)
    zzzzz = Conv3D(1, par['kernel_size'], padding = 'valid', activation = 'tanh')(z5z)

    zzzz = concatenate([zzzzx, zzzzy, zzzzz], axis = -1)

    return zzzz


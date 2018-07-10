'''
Architectures for producing the displacement field
'''
#from keras.models import Model
#from spatial_deformer_net3d import SpatialDeformer3D
from keras.layers import Input, Conv3D, MaxPooling3D, AveragePooling3D, UpSampling3D, multiply, Conv3DTranspose, ZeroPadding3D, Cropping3D, PReLU, concatenate, add, GaussianNoise
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

    #z2_2 = Conv3D(64, (2,2,2), padding = 'same')(z2_2)


    z4 = Conv3DTranspose(64, (2,2,2), strides=2, padding = 'valid')(z2_2)
    z4 = Conv3D(64, (2,2,2), padding = 'same', activation = 'linear')(z4)
#    z4 = PReLU(shared_axes = [4])(z4)
    
    z5 = Conv3DTranspose(32, (2,2,2), strides=2, padding = 'valid')(z4)
    z5 = Conv3D(32, (2,2,2), padding = 'same', activation = 'linear')(z5)
#    z5 = PReLU(shared_axes = [4])(z5)

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

def SDN_ver4(inputs):
    z1_1 = Conv3D(32, (2,2,2), padding = 'same')(inputs)
    z1_2 = Conv3D(32, (2,2,2), strides = 2, padding = 'valid')(z1_1)
    z1_2 = PReLU(shared_axes = [4])(z1_2)

    z2_1 = Conv3D(64, (2,2,2), padding = 'same')(z1_2)
    z2_2 = Conv3D(64, (2,2,2), strides = 2, padding = 'valid')(z2_1)
    z2_2 = PReLU(shared_axes = [4])(z2_2)

    z4 = Conv3DTranspose(64, (2,2,2), strides=2, padding = 'valid')(z2_2)
    #z4 = PReLU(shared_axes = [4])(z4)
    z4 = Conv3D(64, (2,2,2), padding = 'same')(z4)
    #z4 = PReLU()(z4)
    z5 = Conv3DTranspose(32, (2,2,2), strides=2, padding = 'valid')(z4)
    z5 = Conv3D(32, (2,2,2), padding = 'same', activation = 'linear')(z5)
    #z5 = PReLU()(z5)  
    z5 = ZeroPadding3D((2,1,2))(z5)     #Extra padding to make size match
    zzzz = Conv3D(3, (2,2,2), padding = 'valid',
#                      kernel_initializer= 'he_uniform',
#                      bias_initializer = 'he_uniform',
#                      activity_regularizer = l1(0.001),
                      activation = 'tanh')(z5)

    return zzzz


def SDN_ver5(inputs):
    #z1_1 = Conv3D(32, (2,2,2), padding = 'same')(inputs)
    z1_2 = Conv3D(32, (2,2,2), strides = 2, padding = 'valid')(inputs)
    z1_2 = PReLU(shared_axes = [4])(z1_2)

    #z2_1 = Conv3D(64, (2,2,2), padding = 'same')(z1_2)
    z2_2 = Conv3D(64, (2,2,2), strides = 2, padding = 'valid')(z1_2)
    z2_2 = PReLU(shared_axes = [4])(z2_2)

    z4 = Conv3DTranspose(32, (2,2,2), strides=2, padding = 'valid')(z2_2)
    z4 = ZeroPadding3D(((1,0), (0,0), (1,0)))(z4)
    z4 = add([z4, z1_2])
    z4 = PReLU(shared_axes = [4])(z4)
    z4 = Conv3D(32, (2,2,2), padding = 'same')(z4)
   #z4 = PReLU()(z4)

    z5 = Conv3DTranspose(64, (2,2,2), strides=2, padding = 'valid')(z4)
    z5 = Conv3D(64, (2,2,2), padding = 'same', activation = 'linear')(z5)
    #z5 = PReLU()(z5)  
    z5 = ZeroPadding3D(((1,0),(1,0),(1,0)))(z5)     #Extra padding to make size match
    zzzz = Conv3D(3, (2,2,2), padding = 'same',
#                      kernel_initializer= 'he_uniform',
#                      bias_initializer = 'he_uniform',
#                      activity_regularizer = l1(0.001),
                      activation = 'tanh')(z5)

    return zzzz

def SDN_ver6(inputs):
   # z1_1 = Conv3D(32, (2,2,2), padding = 'same')(inputs)
    z1_2 = Conv3D(32, (2,2,2), strides = 2, padding = 'valid')(inputs)
    z1_2 = PReLU(shared_axes = [4])(z1_2)

    #z2_1 = Conv3D(64, (2,2,2), padding = 'same')(z1_2)
    z2_2 = Conv3D(64, (2,2,2), strides = 2, padding = 'valid')(z1_2)
    z2_2 = PReLU(shared_axes = [4])(z2_2)

    z4 = Conv3DTranspose(64, (2,2,2), strides=2, padding = 'valid')(z2_2)
    z4 = PReLU(shared_axes = [4])(z4)
    z4 = Conv3D(64, (2,2,2), padding = 'same')(z4)
    #z4 = PReLU()(z4)
    z5 = Conv3DTranspose(32, (2,2,2), strides=2, padding = 'valid')(z4)
    z5 = Conv3D(32, (2,2,2), padding = 'same', activation = 'linear')(z5)
    #z5 = PReLU()(z5)  
    z5 = ZeroPadding3D((2,1,2))(z5)     #Extra padding to make size match
    zzzz = Conv3D(3, (2,2,2), padding = 'valid',
#                      kernel_initializer= 'he_uniform',
#                      bias_initializer = 'he_uniform',
#                      activity_regularizer = l1(0.001),
                      activation = 'tanh')(z5)

    return zzzz


def SDN_ver7(inputs):  

    z1_2 = Conv3D(32, (3,3,3), strides = 2, padding = 'valid')(inputs)
    z1_2 = PReLU(shared_axes = [4])(z1_2)


    z2_2 = Conv3D(64, (2,2,2), strides = 2, padding = 'valid')(z1_2)
    z2_2 = PReLU(shared_axes = [4])(z2_2)


    z4 = Conv3DTranspose(64, (2,2,2), strides=2, padding = 'valid')(z2_2)
    z4 = Conv3D(64, (2,2,2), padding = 'same')(z4)

    z5 = Conv3DTranspose(32, (3,3,3), strides=2, padding = 'valid')(z4)
    z5 = Conv3D(32, (3,3,3), padding = 'same', activation = 'linear')(z5)

    z5 = ZeroPadding3D((1,0,1))(z5)     #Extra padding to make size match
    zzzz = Conv3D(3, (2,2,2), padding = 'same',
#                      kernel_initializer= 'he_uniform',
#                      bias_initializer = 'he_uniform',
#                      activity_regularizer = l1(0.001),
                      activation = 'tanh')(z5)

    return zzzz


def SDN_ver8(inputs):  
    z1_1 = Conv3D(32, (2,2,2), padding = 'same')(inputs)
    z1_2 = Conv3D(32, (2,2,2), strides = 2, padding = 'valid')(z1_1)
   # z1_2 = BatchNormalization()(z1_2)

    z2_1 = Conv3D(64, (2,2,2), padding = 'same')(z1_2)
    z2_2 = Conv3D(64, (2,2,2), strides = 2, padding = 'valid')(z2_1)


    z3 = Conv3D(128, (2,2,2), padding = 'same')(z2_2)


    z4 = Conv3DTranspose(64, (2,2,2), strides=(2,2,2), padding = 'valid')(z3)
#    z4 = Conv3D(64, (2,2,2), padding = 'same')(z4)

    z5 = Conv3DTranspose(32, (2,2,2), strides=(2,2,2), padding = 'valid')(z4)
#    z5 = Conv3D(32, (2,2,2), padding = 'same', activation = 'linear')(z5)

    zzzz = Conv3D(3, (2,2,2), padding = 'same',
#                      kernel_initializer= 'he_uniform',
#                      bias_initializer = 'he_uniform',
#                      activity_regularizer = l1(0.001),
                      activation = 'tanh')(z5)

    return zzzz


def SDN_ver9(inputs):  
    #z1_1 = Conv3D(32, (2,2,2), padding = 'same')(inputs)
    z1_2 = Conv3D(32, (2,2,2), strides = 2, padding = 'valid')(inputs)
    z1_2 = PReLU(shared_axes = [1,2,3])(z1_2)
    #z2_1 = Conv3D(64, (2,2,2), padding = 'same')(z1_2)
    z2_2 = Conv3D(64, (2,2,2), strides = 2, padding = 'valid')(z1_2)
    z2_2 = PReLU(shared_axes = [1,2,3])(z2_2)
    #z3 = Conv3D(64, (2,2,2), padding = 'same')(z2_2)
    #z3 = PReLU()(z3)
    z4 = Conv3DTranspose(64, (2,2,2), strides=2, padding = 'valid')(z2_2)
    z4 = PReLU(shared_axes = [1,2,3])(z4)
    z4 = Conv3D(64, (2,2,2), padding = 'same')(z4)
    z4 = PReLU(shared_axes = [1,2,3])(z4)
    z5 = Conv3DTranspose(32, (2,2,2), strides=2, padding = 'valid')(z4)
    z5 = PReLU(shared_axes = [1,2,3])(z5)
    z5 = Conv3D(32, (2,2,2), padding = 'same', activation = 'linear')(z5)
    z5 = PReLU(shared_axes = [1,2,3])(z5)
    z5 = ZeroPadding3D((2,1,2))(z5)     #Extra padding to make size match
    zzzz = Conv3D(3, (2,2,2), padding = 'valid',
#                      kernel_initializer= 'he_uniform',
#                      bias_initializer = 'he_uniform',
#                      activity_regularizer = l1(0.001),
                      activation = 'tanh')(z5)
    return zzzz

def SDN_ver10(inputs):    
    z1_2 = Conv3D(32, (2,2,2), strides = 2, padding = 'valid')(inputs)
    z1_2 = PReLU(shared_axes = [4])(z1_2)

    z2_2 = Conv3D(64, (2,2,2), strides = 2, padding = 'valid')(z1_2)
    z2_2 = PReLU(shared_axes = [4])(z2_2)

    z3 = Conv3D(128, (2,2,2), strides = 2, padding = 'valid')(z2_2)
    z3 = PReLU(shared_axes= [4])(z3)

    z3 = Conv3DTranspose(128, (2,2,2), strides=2, padding = 'valid')(z3)
    #z3 = Conv3D(64, (2,2,2), padding = 'same')(z3)

    z4 = Conv3DTranspose(64, (2,2,2), strides=2, padding = 'valid')(z3)
    z4 = Conv3D(64, (2,2,2), padding = 'same')(z4)
    #z4 = PReLU()(z4)
    z5 = Conv3DTranspose(32, (2,2,2), strides=2, padding = 'valid')(z4)
    z5 = Conv3D(32, (2,2,2), padding = 'same', activation = 'linear')(z5)
    #z5 = PReLU()(z5)  
    z5 = ZeroPadding3D((2,3,2))(z5)     #Extra padding to make size match
    zzzz = Conv3D(3, (2,2,2), padding = 'valid', activation = 'tanh')(z5)

    return zzzz

def incept(inputs, num_channel, activation = 'linear'):
    z1 = Conv3D(num_channel, (2,2,2), padding = 'same', activation = activation)(inputs)
    z2 = Conv3D(num_channel, (5,5,5), padding = 'same', activation = activation)(inputs)
    z3 = Conv3D(num_channel, (7,7,7), padding = 'same', activation = activation)(inputs)
    z4 = Conv3D(num_channel, (11,11,11), padding = 'same', activation = activation)(inputs)
   
    z = concatenate([z1, z2, z3, z4])
    z = PReLU(shared_axes = [4])(z)
    return z


def incept2(inputs, num_channel, activation = 'relu'):
    '''
    Google's Inception-like
    '''
    z1 = Conv3D(num_channel, (1,1,1), padding = 'same', activation = activation)(inputs)
    z2 = Conv3D(num_channel, (3,3,3), padding = 'same', activation = activation)(inputs)
    z3 = Conv3D(num_channel, (5,5,5), padding = 'same', activation = activation)(inputs)
    z4 = MaxPooling3D((3,3,3), (1,1,1), padding = 'same')(inputs)  # which pooling?
    z4 = Conv3D(num_channel, (1,1,1), padding = 'same', activation = activation)(z4)
    
    z = concatenate([z3, z2, z4, z1])
    #z = PReLU(shared_axes = [4])(z)
    return z


def incept3(inputs, num_channel, activation = 'relu'):
    '''
    Google's Inception-like with dimension reduction
    '''
    z1 = Conv3D(num_channel, (1,1,1), padding = 'same', activation = activation)(inputs)
    z2 = Conv3D(num_channel, (1,1,1), padding = 'same', activation = activation)(inputs)
    z2 = Conv3D(num_channel, (3,3,3), padding = 'same', activation = activation)(z2)

    z3 = Conv3D(num_channel, (1,1,1), padding = 'same', activation = activation)(inputs)
    z3 = Conv3D(num_channel, (5,5,5), padding = 'same', activation = activation)(z3)

    z4 = MaxPooling3D((3,3,3), (1,1,1),padding = 'same')(inputs)
    z4 = Conv3D(num_channel, (1,1,1), padding = 'same', activation = activation)(z4)

    z = concatenate([z3, z2, z4, z1])
    #z = PReLU(shared_axes = [4])(z)
    return z

def incept4(inputs, num_channel, activation = 'relu'):
    '''
    Google's Inception-like with dimension reduction
    '''
    z1 = Conv3D(num_channel, (1,1,1), padding = 'same', activation = activation)(inputs)
    z2 = Conv3D(num_channel, (3,3,3), padding = 'same', activation = activation)(z1)

    z3 = Conv3D(num_channel, (5,5,5), padding = 'same', activation = activation)(z1)

    z4 = AveragePooling3D((3,3,3), (1,1,1),padding = 'same')(inputs)
    z4 = Conv3D(num_channel, (1,1,1), padding = 'same', activation = activation)(z4)

    z = concatenate([z3, z2, z4, z1])
    #z = PReLU(shared_axes = [4])(z)
    return z


def SDN_ver11(inputs):  
    
    z1_1 = incept2(inputs, 8)
    z1_2 = Conv3D(32, (2,2,2), strides = 2, padding = 'valid')(z1_1)
    z1_2 = PReLU(shared_axes = [4])(z1_2)

    
    z2_2 = Conv3D(64, (2,2,2), strides = 2, padding = 'valid')(z1_2)
    z2_2 = PReLU(shared_axes = [4])(z2_2)

    
    z4 = Conv3DTranspose(64, (2,2,2), strides=2, padding = 'valid')(z2_2)
    z4 = Conv3D(64, (2,2,2), padding = 'same', activation = 'linear')(z4)

    z5 = Conv3DTranspose(32, (2,2,2), strides=2, padding = 'valid')(z4)
    z5 = Conv3D(32, (2,2,2), padding = 'same', activation = 'linear')(z5)

    z5 = ZeroPadding3D((2,1,2))(z5)     #Extra padding to make size match
    zzzz = Conv3D(3, (2,2,2), padding = 'valid',
                      activation = 'tanh')(z5)
    return zzzz


def SDN_ver12(inputs):  
    
    z1_1 = incept(inputs, 8)
    z1_2 = Conv3D(32, (2,2,2), strides = 2, padding = 'valid')(z1_1)
    z1_2 = PReLU(shared_axes = [4])(z1_2)

    
    z2_2 = Conv3D(64, (2,2,2), strides = 2, padding = 'valid')(z1_2)
    z2_2 = PReLU(shared_axes = [4])(z2_2)

    
    z4 = Conv3DTranspose(64, (2,2,2), strides=2, padding = 'valid')(z2_2)
    z4 = Conv3D(64, (2,2,2), padding = 'same', activation = 'linear')(z4)

    z5 = Conv3DTranspose(32, (2,2,2), strides=2, padding = 'valid')(z4)
    z5 = incept4(z5, 8, 'linear')
    z5 = Conv3D(32, (2,2,2), padding = 'same', activation = 'linear')(z5) 

    z5 = ZeroPadding3D((2,1,2))(z5)     #Extra padding to make size match
    zzzz = Conv3D(3, (2,2,2), padding = 'valid',
                      activation = 'tanh')(z5)
    return zzzz

def unet(x_in, enc_nf, dec_nf, full_size=True):
    """
    unet network for voxelmorph 

    Args:
        vol_size: volume size. e.g. (256, 256, 256)
        enc_nf: encoder filters. right now it needs to be to 1x4.
            e.g. [16,32,32,32]
            TODO: make this flexible.
        dec_nf: encoder filters. right now it's forced to be 1x7.
            e.g. [32,32,32,32,8,8,3]
            TODO: make this flexible.
        full_size

    """

    x_in = ZeroPadding3D(((3,2), (2,1), (3,2)))(x_in)
    
    # down-sample path.
    x0 = myConv(x_in, enc_nf[0], 2)  # 80x96x112
    x1 = myConv(x0, enc_nf[1], 2)  # 40x48x56
    x2 = myConv(x1, enc_nf[2], 2)  # 20x24x28
    x3 = myConv(x2, enc_nf[3], 2)  # 10x12x14

    # up-sample path.
    x = myConv(x3, dec_nf[0])
    x = UpSampling3D()(x)
    #x2 = ZeroPadding3D(((1,0), (0,0), (1,0)))(x2)
    x = concatenate([x, x2])
    x = myConv(x, dec_nf[1])
    x = UpSampling3D()(x)
    #x1 = ZeroPadding3D(((1,0), (0,0), (1,0)))(x1)
    x = concatenate([x, x1])
    x = myConv(x, dec_nf[2])
    x = UpSampling3D()(x)
    #x0 = ZeroPadding3D(((1,1), (1,0), (1,1)))(x0)
    x = concatenate([x, x0])
    x = myConv(x, dec_nf[3])
    x = myConv(x, dec_nf[4])

    if full_size:
        x = UpSampling3D()(x)
        x = concatenate([x, x_in])
        x = myConv(x, dec_nf[5])

        # optional convolution
        if (len(dec_nf) == 8):
            x = myConv(x, dec_nf[6])

    # transform the results into a flow.
    flow = Conv3D(dec_nf[-1], kernel_size=3, padding='same',
                  kernel_initializer=RandomNormal(mean=0.0, stddev=1e-5), name='flow')(x)
    flow = Lambda(lambda x: x[:,3:-2, 2:-1, 3:-2, :])(flow)
    
    return flow          

'''
Architectures for producing the displacement field
'''
#from keras.models import Model
#from spatial_deformer_net3d import SpatialDeformer3D
from keras.layers import Input, Conv3D, MaxPooling3D, AveragePooling3D, UpSampling3D, multiply, Conv3DTranspose, ZeroPadding3D, Cropping3D, PReLU, concatenate, add, GaussianNoise, LeakyReLU, Lambda
#import keras.backend as K
from keras.initializers import RandomNormal
import tensorflow as tf

par = {
       'kernel_size': (3,3,3),

       'kernel_strides': 2,

      }

def gaussian_kernel(size: int, mean: float, std: float):
    """Makes 1d gaussian Kernel for convolution."""

    d = tf.distributions.Normal(mean, std)

    vals = d.prob(tf.range(start = -size, limit = size + 1, dtype = tf.float32))

    gauss_kernel2d = tf.einsum('i,j->ij', vals, vals)
    guass_kernel3d = tf.einsum('ij,k->ijk', gauss_kernel2d, vals)
    kernel = guass_kernel3d / tf.reduce_sum(guass_kernel3d)
    kernel = kernel[...,tf.newaxis]
    return tf.stack([kernel, kernel, kernel], 4) # (HWD1C)
    

# Make Gaussian Kernel with desired specs.
def expandedKernel(inputTensor, kernel):
    #this considers data_format = 'channels_last'
    inputChannels = tf.reshape(tf.ones_like(inputTensor[0,0,0,0,:]),(1,1,1,-1,1))
    #if you're using 'channels_first', use inputTensor[0,:,0,0] above

    return kernel*inputChannels

def Gaussian3D(image, kernel):
    # Expand dimensions of `gauss_kernel` for `tf.nn.conv3d` signature.
    expandedK = expandedKernel(image, kernel)
                               
    # Convolve.
    return tf.nn.conv3d(image, expandedK, strides=[1, 1, 1, 1, 1], padding="SAME")


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

#    z5 = ZeroPadding3D((1,1,1))(z5)     #Extra padding to make size match
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
  


def incept(inputs, num_channel, activation = 'linear'):
    z1 = Conv3D(num_channel, (2,2,2), padding = 'same', activation = activation)(inputs)
    z2 = Conv3D(num_channel, (5,5,5), padding = 'same', activation = activation)(inputs)
    z3 = Conv3D(num_channel, (7,7,7), padding = 'same', activation = activation)(inputs)
    z4 = Conv3D(num_channel, (11,11,11), padding = 'same', activation = activation)(inputs)
   
    z = concatenate([z4, z3, z2, z1])
    z = PReLU(shared_axes = [1,2,3])(z)
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
    z = PReLU(shared_axes = [4])(z)
    return z

def incept4(inputs, num_channel, activation = 'PReLU'):
    '''
    Google's Inception-like with dimension reduction
    '''
    z1 = Conv3D(num_channel, (1,1,1), padding = 'same')(inputs)
    z2 = Conv3D(num_channel, (3,3,3), padding = 'same')(z1)

    z3 = Conv3D(num_channel, (5,5,5), padding = 'same')(z1)

    z4 = AveragePooling3D((3,3,3), (1,1,1),padding = 'same')(inputs)
    z4 = Conv3D(num_channel, (1,1,1), padding = 'same')(z4)

    z = concatenate([z3, z2, z4, z1])
    
    if activation is 'PReLU':
       z = PReLU(shared_axes = [1, 2, 3])(z)
    elif activation is 'LeakyReLU':
       z = LeakyReLU(0.2)(z)

    return z


def SDN_ver11(inputs):  
    
    z1_1 = incept(inputs, 8) #location of it?
    z1_2 = Conv3D(32, par['kernel_size'], strides = 2, padding = 'valid')(z1_1)
    z1_2 = PReLU(shared_axes = [4])(z1_2)

    
    z2_2 = Conv3D(64, par['kernel_size'], strides = 2, padding = 'valid')(z1_2)
    z2_2 = PReLU(shared_axes =[4])(z2_2)

#    z2_3 = incept4(z2_2, 16, 'linear')
#    z2_2 = add([z2_2, z2_3])

    z4 = Conv3DTranspose(64, par['kernel_size'], strides=2, padding = 'valid')(z2_2)
    z4 = Conv3D(64, par['kernel_size'], padding = 'same', activation = 'linear')(z4)
    
    #z4 = add([z4, z1_2])

    z5 = Conv3DTranspose(32, par['kernel_size'], strides=2, padding = 'valid')(z4)
    z5 = Conv3D(32, par['kernel_size'], padding = 'same', activation = 'linear')(z5)
    
    #z5 = add([z5, z1_1])
    z5 = ZeroPadding3D(((0,1),(0,1),(0,1)))(z5)     #Extra padding to make size match
    zzzz = Conv3D(3, par['kernel_size'], padding = 'same', kernel_initializer=RandomNormal(mean=0.0, stddev=1e-2), activation = 'linear')(z5)
    return zzzz


def SDN_incept1(inputs):

    z1_1 = Conv3D(32, par['kernel_size'], padding = 'same')(inputs)
    #z1_1 = PReLU(shared_axes = [1,2,3])(z1_1)
    z1_2 = Conv3D(32, par['kernel_size'], strides = 2, padding = 'valid')(z1_1)
    z1_2 = PReLU(shared_axes = [1,2,3])(z1_2)

    z2_1 = Conv3D(32, par['kernel_size'], padding = 'same')(z1_2)
    #z2_1 = PReLU(shared_axes = [1,2,3])(z2_1)
    z2_2 = Conv3D(32, par['kernel_size'], strides = 2, padding = 'valid')(z2_1)
    z2_2 = PReLU(shared_axes =[1,2,3])(z2_2)
    
    z3 = Conv3D(32, (2,2,2), padding = 'same')(z2_2)
    #z3 = PReLU(shared_axes = [1,2,3])(z3)
    
    z2_3 = incept4(z2_2, 8, 'LeakyReLU')
    z3 = add([z3, z2_3])

    z4 = Conv3DTranspose(32, par['kernel_size'], strides=2, padding = 'valid')(z3)
    #z4 = PReLU(shared_axes = [1,2,3])(z4)
    z4 = Conv3D(32, par['kernel_size'], padding = 'same', activation = 'linear')(z4)
    z4 = PReLU(shared_axes = [1,2,3])(z4)
    
    z1_3 = incept4(z1_2, 8, 'LeakyReLU')
    z4 = add([z4, z1_3])

    z5 = Conv3DTranspose(32, par['kernel_size'], strides=2, padding = 'valid')(z4)
    #z5 = PReLU(shared_axes = [1,2,3])(z5)
    z5 = Conv3D(32, par['kernel_size'], padding = 'same', activation = 'linear')(z5)
    z5 = PReLU(shared_axes = [1,2,3])(z5)
    z5 = ZeroPadding3D(((0,1),(0,1),(0,1)))(z5)   #Extra padding to make size match
    
    z1_4 = incept4(z1_1, 8, 'LeakyReLU')
    z5 = add([z5, z1_4])

    zzzz = Conv3D(3, par['kernel_size'], padding = 'same', kernel_initializer=RandomNormal(mean=0.0, stddev=1e-2), activation = 'linear')(z5)
    
    return zzzz

def SDN_incept(inputs):

#    z1_1 = Conv3D(32, par['kernel_size'], padding = 'same')(inputs)
    #z1_1 = PReLU(shared_axes = [1,2,3])(z1_1)
    z1_1 = incept(inputs,8)
    z1_2 = Conv3D(32, par['kernel_size'], strides = 2, padding = 'valid')(z1_1)
    z1_2 = PReLU(shared_axes = [1,2,3])(z1_2)

    z2_1 = Conv3D(32, par['kernel_size'], padding = 'same')(z1_2)
    #z2_1 = PReLU(shared_axes = [1,2,3])(z2_1)
    z2_2 = Conv3D(32, par['kernel_size'], strides = 2, padding = 'valid')(z2_1)
    z2_2 = PReLU(shared_axes =[1,2,3])(z2_2)

    z3 = Conv3D(32, (2,2,2), padding = 'same')(z2_2)
    #z3 = PReLU(shared_axes = [1,2,3])(z3)

    z3 = add([z3, z2_2])

    z4 = Conv3DTranspose(32, par['kernel_size'], strides=2, padding = 'valid')(z3)
    #z4 = PReLU(shared_axes = [1,2,3])(z4)
    z4 = Conv3D(32, par['kernel_size'], padding = 'same', activation = 'linear')(z4)
    z4 = PReLU(shared_axes = [1,2,3])(z4)
    z4 = add([z4, z1_2])

    z5 = Conv3DTranspose(32, par['kernel_size'], strides=2, padding = 'valid')(z4)
    #z5 = PReLU(shared_axes = [1,2,3])(z5)
    z5 = Conv3D(32, par['kernel_size'], padding = 'same', activation = 'linear')(z5)
    z5 = PReLU(shared_axes = [1,2,3])(z5)
    z5 = ZeroPadding3D(((0,1),(0,1),(0,1)))(z5)   #Extra padding to make size match

    z5 = add([z5, z1_1])
    zzzz = Conv3D(3, par['kernel_size'], padding = 'same', kernel_initializer=RandomNormal(mean=0.0, stddev=1e-2), activation = 'linear')(z5)

    return zzzz

def SDN_incept_G(inputs):
    z = SDN_incept(inputs)
    Gfilter = lambda x: Gaussian3D(x, gaussian_kernel(2, 0.0, 1.0))
    z = Lambda(Gfilter)(z) 
    return z

def Blockdown(inputs, nfilters=32):
    z = Conv3D(nfilters, par['kernel_size'], padding = 'same')(inputs)
    z = Conv3D(nfilters, par['kernel_size'], strides = 2, padding = 'valid')(z)
    z = PReLU(shared_axes = [1,2,3])(z)
    return z

def Blockup(inputs, nfilters=32):
    z = Conv3DTranspose(nfilters, par['kernel_size'], strides=2, padding = 'valid')(inputs)
    z = Conv3D(nfilters, par['kernel_size'], padding = 'same', activation = 'linear')(z)
    z = PReLU(shared_axes = [1,2,3])(z)
    return z

def SDN_incept4(inputs):
    z1 = Blockdown(inputs)
    z2 = Blockdown(z1)
    z3 = Blockdown(z2)
    z4 = Blockdown(z3)

    z5 = Conv3D(32, (2,2,2), padding = 'same')(z4)
    z5 = add([z5, z4])

    z3t = Blockup(z5)
    z3t = add([z3t, z3])
    z2t = Blockup(z3t)
    z2t = ZeroPadding3D(((0,0),(0,1),(0,0)))(z2t)
    z2t = add([z2t, z2])
    z1t = Blockup(z2t)
    z1t = add([z1t, z1])
    z0t = Blockup(z1t)
    z0t = ZeroPadding3D(((0,1),(0,1),(0,1)))(z0t)
    
    disp = Conv3D(32, par['kernel_size'], padding = 'same')(inputs)
    z0t = add([z0t, disp])

    zzzz = Conv3D(3, par['kernel_size'], padding = 'same', kernel_initializer=RandomNormal(mean=0.0, stddev=1e-2), activation = 'linear')(z0t)

    return zzzz

def SDN_incept2(inputs):

    z1_1 = Conv3D(32, par['kernel_size'], padding = 'same')(inputs)
    z1_1 = LeakyReLU(0.2)(z1_1)
    z1_2 = Conv3D(32, par['kernel_size'], strides = 2, padding = 'valid')(z1_1)
    z1_2 = LeakyReLU(0.2)(z1_2)

    z2_1 = Conv3D(32, par['kernel_size'], padding = 'same')(z1_2)
    z2_1 = LeakyReLU(0.2)(z2_1)
    z2_2 = Conv3D(32, par['kernel_size'], strides = 2, padding = 'valid')(z2_1)
    z2_2 = LeakyReLU(0.2)(z2_2)

    z3 = Conv3D(32, (2,2,2), padding = 'same')(z2_2)
    z3 = LeakyReLU(0.2)(z3)
    
    z2_3 = incept4(z2_2, 8, 'LeakyReLU')
    z3 = add([z3, z2_3])

    z4 = Conv3DTranspose(32, par['kernel_size'], strides=2, padding = 'valid')(z3)
    z4 = LeakyReLU(0.2)(z4)
    z4 = Conv3D(32, par['kernel_size'], padding = 'same', activation = 'linear')(z4)
    z4 = LeakyReLU(0.2)(z4)

    z1_3 = incept4(z1_2, 8, 'LeakyReLU')
    z4 = add([z4, z1_3])

    z5 = Conv3DTranspose(32, par['kernel_size'], strides=2, padding = 'valid')(z4)
    z5 = LeakyReLU(0.2)(z5)
    z5 = Conv3D(32, par['kernel_size'], padding = 'same', activation = 'linear')(z5)
    z5 = LeakyReLU(0.2)(z5)
    z5 = ZeroPadding3D(((0,1),(0,1),(0,1)))(z5)   #Extra padding to make size match
    
    z1_4 = incept4(z1_1, 8,'LeakyReLU')
    z5 = add([z5, z1_4])
    zzzz = Conv3D(3, par['kernel_size'], padding = 'same', kernel_initializer=RandomNormal(mean=0.0, stddev=1e-2), activation = 'linear')(z5)

    return zzzz

def MyBlockDown(inputs, nfilters=32):
    z = Conv3D(nfilters, par['kernel_size'], padding = 'same')(inputs)
    z = LeakyReLU(0.2)(z)
    z = Conv3D(nfilters, par['kernel_size'], strides = 2, padding = 'valid')(z)
    z = LeakyReLU(0.2)(z)

    return z

def MyBlockUp(inputs, nfilters=32):
    z = Conv3D(nfilters, par['kernel_size'], padding = 'same')(inputs)
    z = LeakyReLU(0.2)(z)
    z = Conv3DTranspose(nfilters, par['kernel_size'], strides=2, padding = 'valid')(z)
    z = LeakyReLU(0.2)(z)
    return z

def SDN_incept3(inputs):
    z1 = MyBlockDown(inputs, 16)
    z2 = MyBlockDown(z1)
    z3 = MyBlockDown(z2)
    z4 = MyBlockDown(z3)

    z3t = MyBlockUp(z4)
#    z3t = ZeroPadding3D(((0,1),(0,0),(0,1)))(z3t)
    z3t = add([z3t, incept4(z3,8,'LeakyReLU')])
    z2t = MyBlockUp(z3t)
    z2t = ZeroPadding3D(((0,0),(0,1),(0,0)))(z2t)
    z2t = add([z2t, incept4(z2,8,'LeakyReLU')])
    z1t = MyBlockUp(z2t)
#    z1t = ZeroPadding3d(((0,1),(0,1),(0,1)))(z1t)
    z1t = add([z1t, incept4(z1,8,'LeakyReLU')])
    z0t = MyBlockUp(z1t)
#    z0t = ZeroPadding3d(((1,1),(1,1),(1,1)))(z0t)

    disp = Conv3D(16, par['kernel_size'], padding = 'same')(z0t)
    disp = LeakyReLU(0.2)(disp)
    disp = ZeroPadding3D(((0,1),(0,1),(0,1)))(disp)
    disp = add([disp, incept4(inputs, 4, 'LeakyReLU')])
    disp = Conv3D(3, par['kernel_size'], padding = 'same', kernel_initializer=RandomNormal(mean=0.0, stddev=1e-2), activation = 'linear')(disp)
    
    return disp


def myConv(x_in, nf, strides=1):
    """
    specific convolution module including convolution followed by leakyrelu
    """

    x_out = Conv3D(nf, kernel_size=3, padding='same', kernel_initializer='he_normal', strides=strides)(x_in)
    x_out = LeakyReLU(0.2)(x_out)
    return x_out

'''
Modified from  https://github.com/balakg/voxelmorph
'''
def Unet(x_in, vol_size, enc_nf, dec_nf, full_size=True):
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
    flow = Lambda(lambda x: x[:,3:-2, 2:-1, 3:-2, :])(flow) #truncate to match size
    
    return flow          

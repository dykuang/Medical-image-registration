# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 21:45:36 2018

@author: Dongyang

Just a script containing help functions for realizing the 

flip transformation 
"""

import tensorflow as tf

def Jac(x):
    batchsize = tf.shape(x)[0]
    height = tf.shape(x)[1]
    width = tf.shape(x)[2]
    depth = tf.shape(x)[3]
    num_channel = tf.shape(x)[4]
    num_voxel = (height-1)*(width-1)*(depth-1)

    dx = tf.reshape(x[:,1:,:-1,:-1,:]-x[:,:-1,:-1,:-1,:], [batchsize, num_voxel, num_channel])
    dy = tf.reshape(x[:,:-1,1:,:-1,:]-x[:,:-1,:-1,:-1,:], [batchsize, num_voxel, num_channel])
    dz = tf.reshape(x[:,:-1,:-1,1:,:]-x[:,:-1,:-1,:-1,:], [batchsize, num_voxel, num_channel])
    J = tf.stack([dx, dy, dz], 3)
    return tf.reshape(J, [batchsize, height-1, width-1, depth-1, 3, 3])

def Jac_5(x):
    batchsize = tf.shape(x)[0]
    height = tf.shape(x)[1]
    width = tf.shape(x)[2]
    depth = tf.shape(x)[3]
    num_channel = tf.shape(x)[4]
    num_voxel = (height-4)*(width-4)*(depth-4)

    dx = tf.reshape((x[:-4,2:-2,2:-2,:]-8*x[1:-3,2:-2,2:-2,:] + 8*x[3:-1,2:-2,2:-2,:] - x[4:,2:-2,2:-2,:])/12.0, [num_voxel, num_channel])
    dy = tf.reshape((x[2:-2,:-4,2:-2,:]-8*x[2:-2,1:-3,2:-2,:] + 8*x[2:-2,3:-1,2:-2,:] - x[2:-2,4:,2:-2,:])/12.0, [num_voxel, num_channel])
    dz = tf.reshape((x[2:-2,2:-2,:-4,:]-8*x[2:-2,2:-2,1:-3,:] + 8*x[2:-2,2:-2,3:-1,:] - x[2:-2,2:-2,4:,:])/12.0, [num_voxel, num_channel])
    J = tf.stack([dx, dy, dz], 3)
    return tf.reshape(J, [batchsize, height-4, width-4, depth-4, 3, 3])

def Pushforward(x, Dh, h_inv):
    batchsize = tf.shape(x)[0]
    height = tf.shape(x)[1]
    width = tf.shape(x)[2]
    depth = tf.shape(x)[3]
    num_channel = tf.shape(x)[4]
    num_voxel = (height-1)*(width-1)*(depth-1)
  
    yTrue_f = tf.reshape(h_inv, (batchsize, height*width*depth, num_channel))
    yTrue_f = tf.transpose(yTrue_f, (0, 2, 1))
    x_s = tf.slice(yTrue_f, [0, 0, 0], [-1, 1, -1])
    y_s = tf.slice(yTrue_f, [0, 1, 0], [-1, 1, -1])
    z_s = tf.slice(yTrue_f, [0, 2, 0], [-1, 1, -1])
    X = tf.reshape(x_s, [-1])
    Y = tf.reshape(y_s, [-1])
    Z = tf.reshape(z_s, [-1])
    yPred_r = interpolate(x, Y, Z, X, (height, width, depth))

    return tf.einsum('abcdef,abcdf->abcde', Dh, yPred_r[:,:-1,:-1,:-1,:])


def repeat(x, num_repeats): # copy along the second dimension, each row is a copy of an index
        ones = tf.ones((1, num_repeats), dtype='int32')
        x = tf.reshape(x, shape=(-1,1))
        x = tf.matmul(x, ones)
        return tf.reshape(x, [-1])


def interpolate(image, x, y, z, output_size):  # tri-linear interpolation y-x-z: height-width-depth. !!
        batch_size = tf.shape(image)[0]
        height = tf.shape(image)[1]
        width = tf.shape(image)[2]
        depth = tf.shape(image)[3]
        num_channels = tf.shape(image)[4]

        x = tf.cast(x , dtype='float32')
        y = tf.cast(y , dtype='float32')
        z = tf.cast(z , dtype='float32')

        output_height = output_size[0]
        output_width  = output_size[1]
        output_depth = output_size[2]

        x0 = tf.cast(tf.floor(x), 'int32')
        x1 = x0 + 1
        y0 = tf.cast(tf.floor(y), 'int32')
        y1 = y0 + 1
        z0 = tf.cast(tf.floor(z), 'int32')
        z1 = z0 + 1
        
        max_z = tf.cast(depth - 1,  dtype='int32')
        max_y = tf.cast(height - 1, dtype='int32')
        max_x = tf.cast(width - 1,  dtype='int32')
        
        zero = tf.zeros([], dtype='int32')

        x0 = tf.clip_by_value(x0, zero, max_x)
        x1 = tf.clip_by_value(x1, zero, max_x)
        y0 = tf.clip_by_value(y0, zero, max_y)
        y1 = tf.clip_by_value(y1, zero, max_y)
        z0 = tf.clip_by_value(z0, zero, max_z)
        z1 = tf.clip_by_value(z1, zero, max_z)

        
        xzslice_dimensions = width*depth
        flat_image_dimensions = xzslice_dimensions*height
        pixels_batch = tf.range(batch_size)*flat_image_dimensions
        flat_output_dimensions = output_height*output_width*output_depth
        base = repeat(pixels_batch, flat_output_dimensions)
        
        
        #get indices for the eight corners       
        
        # the slice (depth)
        base_y0 = base + y0*xzslice_dimensions
        base_y1 = base + y1*xzslice_dimensions        
        
        # row in each slice
        base_00 = base_y0 + x0*depth
        base_01 = base_y0 + x1*depth
        base_10 = base_y1 + x0*depth
        base_11 = base_y1 + x1*depth
 
        # each indices
        indices_000 = base_00 + z0
        indices_001 = base_00 + z1
        indices_010 = base_01 + z0
        indices_011 = base_01 + z1
        indices_100 = base_10 + z0
        indices_101 = base_10 + z1
        indices_110 = base_11 + z0
        indices_111 = base_11 + z1             

        flat_image = tf.reshape(image, shape=(-1, num_channels))
        flat_image = tf.cast(flat_image, dtype='float32')
        
        pixel_values_000 = tf.gather(flat_image, indices_000)
        pixel_values_001 = tf.gather(flat_image, indices_001)
        pixel_values_010 = tf.gather(flat_image, indices_010)
        pixel_values_011 = tf.gather(flat_image, indices_011)          
        pixel_values_100 = tf.gather(flat_image, indices_100)
        pixel_values_101 = tf.gather(flat_image, indices_101)
        pixel_values_110 = tf.gather(flat_image, indices_110)
        pixel_values_111 = tf.gather(flat_image, indices_111)       
        
        
        x0 = tf.cast(x0, 'float32')
        x1 = tf.cast(x1, 'float32')
        y0 = tf.cast(y0, 'float32')
        y1 = tf.cast(y1, 'float32')
        z0 = tf.cast(z0, 'float32')
        z1 = tf.cast(z1, 'float32')
        
        # must be careful here!!
        vol_000 = tf.expand_dims((y1-y)*(x1-x)*(z1-z) , 1)
        vol_001 = tf.expand_dims((y1-y)*(x1-x)*(z-z0) , 1)
        vol_010 = tf.expand_dims((y1-y)*(x-x0)*(z1-z) , 1)
        vol_011 = tf.expand_dims((y1-y)*(x-x0)*(z-z0) , 1)
        vol_100 = tf.expand_dims((y-y0)*(x1-x)*(z1-z) , 1)
        vol_101 = tf.expand_dims((y-y0)*(x1-x)*(z-z0) , 1)
        vol_110 = tf.expand_dims((y-y0)*(x-x0)*(z1-z) , 1)
        vol_111 = tf.expand_dims((y-y0)*(x-x0)*(z-z0) , 1)
       
        
        output = tf.add_n([vol_000*pixel_values_000,
                           vol_001*pixel_values_001,
                           vol_010*pixel_values_010,
                           vol_011*pixel_values_011,
                           vol_100*pixel_values_100,
                           vol_101*pixel_values_101,
                           vol_110*pixel_values_110,
                           vol_111*pixel_values_111,
                           ])
     
        return output

def meshgrid(height, width, depth):
        x_linspace = tf.linspace(-1., 1., width)
        y_linspace = tf.linspace(-1., 1., height)
        z_linspace = tf.linspace(-1., 1., depth)
        
        x_coordinates, y_coordinates, z_coordinates = tf.meshgrid(x_linspace, y_linspace, z_linspace)
        
        x_coordinates = tf.reshape(x_coordinates, [-1])
        y_coordinates = tf.reshape(y_coordinates, [-1])
        z_coordinates = tf.reshape(z_coordinates, [-1])

        indices_grid = tf.concat([x_coordinates, y_coordinates, z_coordinates], 0)
                                 
        return indices_grid

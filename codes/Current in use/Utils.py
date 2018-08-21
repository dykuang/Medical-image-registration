# -*- coding: utf-8 -*-
"""
Created on Thu May  3 14:40:50 2018

@author: Dongyang
This script will contain some utility functions
"""

import numpy as np

#===============================================================================
# Calculate the Jacobian
#===============================================================================
def Jac(x):

    height, width, depth, num_channel = x.shape
    num_voxel = (height-1)*(width-1)*(depth-1)

    dx = np.reshape(x[1:,:-1,:-1,:]-x[:-1,:-1,:-1,:], [num_voxel, num_channel])
    dy = np.reshape(x[:-1,1:,:-1,:]-x[:-1,:-1,:-1,:], [num_voxel, num_channel])
    dz = np.reshape(x[:-1,:-1,1:,:]-x[:-1,:-1,:-1,:], [num_voxel, num_channel])
    J = np.stack([dx, dy, dz], 2)
    return np.reshape(J, [height-1, width-1, depth-1, 3, 3])

def Jac_2nd(x):

    height, width, depth, num_channel = x.shape
    num_voxel = (height-2)*(width-2)*(depth-2)

    dx = np.reshape(0.5*(x[2:,1:-1,1:-1,:]-x[:-2,1:-1,1:-1,:]), [num_voxel, num_channel])
    dy = np.reshape(0.5*(x[1:-1,2:,1:-1,:]-x[1:-1,:-2,1:-1,:]), [num_voxel, num_channel])
    dz = np.reshape(0.5*(x[1:-1,1:-1,2:,:]-x[1:-1,1:-1,:-2,:]), [num_voxel, num_channel])
    J = np.stack([dx, dy, dz], 2)
    return np.reshape(J, [height-2, width-2, depth-2, 3, 3])

def Jac_5(x):

    height, width, depth, num_channel = x.shape
    num_voxel = (height-4)*(width-4)*(depth-4)

    dx = np.reshape((x[:-4,2:-2,2:-2,:]-8*x[1:-3,2:-2,2:-2,:] + 8*x[3:-1,2:-2,2:-2,:] - x[4:,2:-2,2:-2,:])/12.0, [num_voxel, num_channel])
    dy = np.reshape((x[2:-2,:-4,2:-2,:]-8*x[2:-2,1:-3,2:-2,:] + 8*x[2:-2,3:-1,2:-2,:] - x[2:-2,4:,2:-2,:])/12.0, [num_voxel, num_channel])
    dz = np.reshape((x[2:-2,2:-2,:-4,:]-8*x[2:-2,2:-2,1:-3,:] + 8*x[2:-2,2:-2,3:-1,:] - x[2:-2,2:-2,4:,:])/12.0, [num_voxel, num_channel])
    J = np.stack([dx, dy, dz], 2)
    return np.reshape(J, [height-4, width-4, depth-4, 3, 3])

#==============================================================================

# Calculate the Determinent of Jacobian of the transformation

#==============================================================================

def Get_Ja(displacement):

    '''

    Calculate the Jacobian value at each point of the displacement map having

    size of b*h*w*d*3 and in the cubic volumn of [-1, 1]^3

    '''

    

    dx = displacement.shape[1]/2.0

    dy = displacement.shape[2]/2.0

    dz = displacement.shape[3]/2.0

    

    D_x = dx*(displacement[:,1:,:-1,:-1,:] - displacement[:,:-1,:-1,:-1,:])

    D_y = dy*(displacement[:,:-1,1:,:-1,:] - displacement[:,:-1,:-1,:-1,:])

    D_z = dz*(displacement[:,:-1,:-1,1:,:] - displacement[:,:-1,:-1,:-1,:])

    

    D1 = (D_x[...,0]+1)*( (D_y[...,1]+1)*(D_z[...,2]+1) - D_z[...,1]*D_y[...,2])

    D2 = (D_x[...,1])*(D_y[...,0]*(D_z[...,2]+1) - D_y[...,2]*D_x[...,0])

    D3 = (D_x[...,2])*(D_y[...,0]*D_z[...,1] - (D_y[...,1]+1)*D_z[...,0])

    

    return np.linalg.norm(D1-D2+D3)


#==============================================================================
# Apply interpolation
#=============================================================================+
def repeat(x, num_repeats): # copy along the second dimension, each row is a copy of an index
    ones = np.ones((1, num_repeats), dtype='int32')
    x = np.reshape(x, (-1,1))
    x = np.matmul(x, ones)
    return np.reshape(x, [-1])

def interpolate(image, x, y, z, output_size):  # tri-linear interpolation y-x-z: height-width-depth. !!
    batch_size = np.shape(image)[0]
    height = np.shape(image)[1]
    width = np.shape(image)[2]
    depth = np.shape(image)[3]
    num_channels = np.shape(image)[4]

#    x = np.cast(x , dtype='float32')
#    y = np.cast(y , dtype='float32')
#    z = np.cast(z , dtype='float32')

    height_float = height
    width_float = width
    depth_float = depth

    output_height = output_size[0]
    output_width  = output_size[1]
    output_depth = output_size[2]

    x = .5*(x + 1.0)*(width_float-1) 
    y = .5*(y + 1.0)*(height_float-1)
    z = .5*(z + 1.0)*(depth_float-1)

    x0 = np.floor(x)
    x1 = x0 + 1
    y0 = np.floor(y)
    y1 = y0 + 1
    z0 = np.floor(z)
    z1 = z0 + 1
      
    max_z = depth - 1
    max_y = height - 1
    max_x = width - 1
    
        
    zero = np.zeros([], dtype='int32')

    x0 = np.clip(x0, zero, max_x).astype('int32')
    x1 = np.clip(x1, zero, max_x).astype('int32')
    y0 = np.clip(y0, zero, max_y).astype('int32')
    y1 = np.clip(y1, zero, max_y).astype('int32')
    z0 = np.clip(z0, zero, max_z).astype('int32')
    z1 = np.clip(z1, zero, max_z).astype('int32')
    
        
    xzslice_dimensions = width*depth
    flat_image_dimensions = xzslice_dimensions*height
    pixels_batch = np.arange(batch_size)*flat_image_dimensions
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

    flat_image = np.reshape(image, (-1, num_channels))
#    flat_image = np.cast(flat_image, dtype='float32')
        
    pixel_values_000 = flat_image[indices_000,:]
    pixel_values_001 = flat_image[indices_001,:]
    pixel_values_010 = flat_image[indices_010,:]
    pixel_values_011 = flat_image[indices_011,:]          
    pixel_values_100 = flat_image[indices_100,:]
    pixel_values_101 = flat_image[indices_101,:]
    pixel_values_110 = flat_image[indices_110,:]
    pixel_values_111 = flat_image[indices_111,:]       
       
        
    # must be careful here!!
    vol_000 = np.expand_dims((y1-y)*(x1-x)*(z1-z),1)
    vol_001 = np.expand_dims((y1-y)*(x1-x)*(z-z0),1)
    vol_010 = np.expand_dims((y1-y)*(x-x0)*(z1-z),1)
    vol_011 = np.expand_dims((y1-y)*(x-x0)*(z-z0),1)
    vol_100 = np.expand_dims((y-y0)*(x1-x)*(z1-z),1)
    vol_101 = np.expand_dims((y-y0)*(x1-x)*(z-z0),1)
    vol_110 = np.expand_dims((y-y0)*(x-x0)*(z1-z),1)
    vol_111 = np.expand_dims((y-y0)*(x-x0)*(z-z0),1)
       
 
    output = vol_000*pixel_values_000+\
             vol_001*pixel_values_001+\
             vol_010*pixel_values_010+\
             vol_011*pixel_values_011+\
             vol_100*pixel_values_100+\
             vol_101*pixel_values_101+\
             vol_110*pixel_values_110+\
             vol_111*pixel_values_111
                  
    return output

def meshgrid(height, width, depth):
    x_linspace = np.linspace(-1., 1., width)
    y_linspace = np.linspace(-1., 1., height)
    z_linspace = np.linspace(-1., 1., depth)
        
    x_coordinates, y_coordinates, z_coordinates = np.meshgrid(x_linspace, y_linspace, z_linspace)
        
    x_coordinates = np.reshape(x_coordinates, [-1])
    y_coordinates = np.reshape(y_coordinates, [-1])
    z_coordinates = np.reshape(z_coordinates, [-1])

    indices_grid = np.stack([x_coordinates, y_coordinates, z_coordinates], 0)
                                 
    return indices_grid

def transform(deformation, input_vol, output_size):
    batch_size = np.shape(input_vol)[0]
#    height = np.shape(input_vol)[1]
#    width = np.shape(input_vol)[2]
#    depth = np.shape(input_vol)[3]
    num_channels = np.shape(input_vol)[4]

#    print(num_channels)
#    width = np.cast(width, dtype='float32')
#    height = np.cast(height, dtype='float32')
#    depth = np.cast(depth, dtype = 'float32')
        
    output_height = output_size[0]
    output_width = output_size[1]
    output_depth = output_size[2]
        
    indices_grid = meshgrid(output_height, output_width, output_depth)

    indices_grid = np.tile(indices_grid, batch_size)
    indices_grid = np.reshape(indices_grid, (batch_size, 3, -1))



    deformation = np.reshape(deformation, (-1, output_height * output_width * output_depth, 3))
    deformation = np.transpose(deformation, (0, 2, 1))

#        transformed_grid = indices_grid
    transformed_grid = indices_grid + deformation # are they of the same shape?
#    print(transformed_grid.shape)
    x_s = transformed_grid[:,0,:] #problem here?
    y_s = transformed_grid[:,1,:] 
    z_s = transformed_grid[:,2,:] 
    x_s_flatten = np.reshape(x_s, [-1])
    y_s_flatten = np.reshape(y_s, [-1])
    z_s_flatten = np.reshape(z_s, [-1])

    transformed_vol = interpolate(input_vol, # modify so it only transform the first channel?
                                  x_s_flatten,
                                  y_s_flatten,
                                  z_s_flatten,
                                  output_size)
    
    transformed_vol = np.reshape(transformed_vol, (batch_size,
                                                   output_height,
                                                   output_width,
                                                   output_depth,
                                                   num_channels)) 
        
    return transformed_vol 

#===============================================================================# Mesurement
#===============================================================================               
from sklearn.metrics import jaccard_similarity_score
def j_score(yTrue, yPred):
     js=[]
     for yT, yP in zip(yTrue, yPred):
          js.append(jaccard_similarity_score((yT>0.1).flatten(), (yP>0.1).flatten()))
     js = np.stack(js)
     return np.mean(js)


def Dice(y_true, y_pred):
     T = (y_true.flatten()>0)
     P = (y_pred.flatten()>0)

     return 2*np.sum(T*P)/(np.sum(T) + np.sum(P))

def count_dice(before, after):
    count_worse=0
    count_equal = 0
    count_better = 0

    for i in range(len(before)):

        if after[i] < before[i]:

            count_worse += 1

        elif after[i] > before[i]:

            count_better += 1

        else:

            count_equal += 1

    return count_worse, count_equal, count_better

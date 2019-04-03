# -*- coding: utf-8 -*-
"""
Created on Tue Jul 10 10:57:33 2018

@author: dykua

This script contains some custom losses
"""
#import keras.backend as K
import tensorflow as tf


def gradientLoss(penalty='l1'):
    #scale = tf.constant([[par['res2']-1, 0, 0], [0, par['res1']-1, 0], [0,0,par['res3']-1]], dtype=tf.float32)
    def loss(y_true, y_pred):
        
        #y_pred_O = tf.einsum('abcde,ef->abcdf',y_pred, scale)*0.5

        dy = tf.abs(y_pred[:, 1:, :, :, :] - y_pred[:, :-1, :, :, :])
        dx = tf.abs(y_pred[:, :, 1:, :, :] - y_pred[:, :, :-1, :, :])
        dz = tf.abs(y_pred[:, :, :, 1:, :] - y_pred[:, :, :, :-1, :])
       
       
        if (penalty == 'l2'):
           dy = dy * dy
           dx = dx * dx
           dz = dz * dz
        d = tf.reduce_mean(dx)+tf.reduce_mean(dy)+tf.reduce_mean(dz)
        return d/3.0

    return loss

def LLoss(penalty='l1'):
    def loss(y_true, y_pred):
        dy = y_pred[:,0,:,:,:]
        dx = y_pred[:,:,0,:,:]
        dz = y_pred[:,:,:,0,:]
        if (penalty == 'l2'):
           dy = dy*dy
           dx = dx*dx
           dz = dz*dz

        d=tf.reduce_mean(dx)+tf.reduce_mean(dy)+tf.reduce_mean(dz)
        return d/3.0
    return loss

def TvsLoss(penalty = 'l1', w1=1, w2=0):
#    scale = 0.5*tf.constant([[par['res2']-1.0, 0, 0], [0, par['res1']-1.0, 0], [0,0,par['res3']-1.0]], dtype=tf.float32)
    def loss(y_true, y_pred):
#        y_pred_O = tf.einsum('abcde,ef->abcdf',y_pred, scale)
        
        xx = y_pred[:,:,:,:,0] #Possible Problems here
        yy = y_pred[:,:,:,:,1]
        zz = y_pred[:,:,:,:,2]

        dy = tf.abs(y_pred[:, 1:, :, :, :] - y_pred[:, :-1, :, :, :])
        dx = tf.abs(y_pred[:, :, 1:, :, :] - y_pred[:, :, :-1, :, :])
        dz = tf.abs(y_pred[:, :, :, 1:, :] - y_pred[:, :, :, :-1, :])
        
        if (penalty == 'l2'):
            dy = dy * dy
            dx = dx * dx
            dz = dz * dz

            yy = yy * yy
            xx = xx * xx
            zz = zz * zz
        
        d = tf.reduce_mean(dx)+tf.reduce_mean(dy)+tf.reduce_mean(dz)
        D = tf.reduce_mean(xx+yy+zz)
        
        return w1*d/3.0+ w2*D
    return loss

def Get_Ja(displacement):

    '''
    Calculate the Jacobian value at each point of the displacement map having

    size of b*h*w*d*3 and in the cubic volumn of [-1, 1]^3

    '''

    D_y = (displacement[:,1:,:-1,:-1,:] - displacement[:,:-1,:-1,:-1,:])

    D_x = (displacement[:,:-1,1:,:-1,:] - displacement[:,:-1,:-1,:-1,:])

    D_z = (displacement[:,:-1,:-1,1:,:] - displacement[:,:-1,:-1,:-1,:])



    D1 = (D_x[...,0]+1)*( (D_y[...,1]+1)*(D_z[...,2]+1) - D_z[...,1]*D_y[...,2])

    D2 = (D_x[...,1])*(D_y[...,0]*(D_z[...,2]+1) - D_y[...,2]*D_x[...,0])

    D3 = (D_x[...,2])*(D_y[...,0]*D_z[...,1] - (D_y[...,1]+1)*D_z[...,0])

    return D1-D2+D3

def NJ_loss(y_true, ypred): 
    '''
    Penalizing locations where Jacobian has negative determinants
    '''
    Neg_Jac = 0.5*(tf.abs(Get_Ja(ypred)) - Get_Ja(ypred))
    return tf.reduce_sum(Neg_Jac)

#==============================================================================
# Modified from  https://github.com/balakg/voxelmorph/losses
#==============================================================================
def cc3D(win=[9,9,9], voxel_weights=None): 
    def loss(I, J):
        I2 = I*I
        J2 = J*J
        IJ = I*J

        filt = tf.ones([win[0], win[1], win[2], 1, 1]) 

        I_sum = tf.nn.conv3d(I, filt, [1, 1, 1, 1, 1], "SAME")
        J_sum = tf.nn.conv3d(J, filt, [1, 1, 1, 1, 1], "SAME")
        I2_sum = tf.nn.conv3d(I2, filt, [1, 1, 1, 1, 1], "SAME")
        J2_sum = tf.nn.conv3d(J2, filt, [1, 1, 1, 1, 1], "SAME")
        IJ_sum = tf.nn.conv3d(IJ, filt, [1, 1, 1, 1, 1], "SAME")

        win_size = win[0]*win[1]*win[2]
        u_I = I_sum/win_size
        u_J = J_sum/win_size

        cross = IJ_sum - u_J*I_sum - u_I*J_sum + u_I*u_J*win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I*u_I*win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J*u_J*win_size

        cc = cross*cross / (I_var*J_var+1e-5)

        # if(voxel_weights is not None):
        #	cc = cc * voxel_weights

        #return -tf.log(tf.reduce_mean(cc))
       # return 1/tf.reduce_mean(cc)-1
        return 1-tf.reduce_mean(cc)
       # return (1-tf.reduce_mean(cc))**2
       # return 1/(tf.reduce_mean(cc)+1+1e-5)-0.5
    return loss

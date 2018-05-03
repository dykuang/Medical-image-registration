# -*- coding: utf-8 -*-
"""
Created on Thu May  3 14:40:50 2018

@author: Dongyang

This script will contain some utility functions
"""

import matplotlib.pyplot as plt
import numpy as np

#==============================================================================
# a multi_slice viewer from
# Modified from datacamp: https://www.datacamp.com/community/tutorials/matplotlib-3d-volumetric-data
#==============================================================================
def remove_keymap_conflicts(new_keys_set):
    for prop in plt.rcParams:
        if prop.startswith('keymap.'):
            keys = plt.rcParams[prop]
            remove_list = set(keys) & new_keys_set
            for key in remove_list:
                keys.remove(key)

def multi_slice_viewer(volume, axis = 0):
    remove_keymap_conflicts({'j', 'k'})
    fig, ax = plt.subplots()

    if axis == 1:
         ax.volume = np.moveaxis(volume, 0, -1)
    elif axis == 2:
         ax.volume = np.moveaxis(volume, [0, 1], [-1, -2])
    else:
         ax.volume = volume
    
    ax.index = volume.shape[0] // 2
    ax.imshow(ax.volume[ax.index])
    fig.canvas.mpl_connect('key_press_event', process_key) # use lambda to pass extra arguments


def process_key(event):
    fig = event.canvas.figure
    ax = fig.axes[0]
    if event.key == 'j':
        previous_slice(ax)
    elif event.key == 'k':
        next_slice(ax)
    fig.canvas.draw()

def previous_slice(ax):
    volume = ax.volume
    ax.index = (ax.index - 1) % volume.shape[0]  # wrap around using %
    ax.images[0].set_array(volume[ax.index])
    print(ax.index)

def next_slice(ax):
    volume = ax.volume
    ax.index = (ax.index + 1) % volume.shape[0]
    ax.images[0].set_array(volume[ax.index])
    print(ax.index)


    
'''
Grid Visualization
'''
def vis_grid(disp, direct = 2): # xy is of shape h*w*2
     
     w, h= np.shape(disp)[0], np.shape(disp)[1]
     
     x = np.linspace(-1., 1., w)
     y = np.linspace(-1., 1., h)
     
     xx , yy = np.meshgrid(x, y)
     
     xy = np.stack([xx,yy], 2) + disp
     
     plt.figure()
     
     if direct == 0: #Only plot the x-direction
          for row in range(w):
               x, y = xy[row,:, 0], yy[row,:]       
               plt.plot(x,y, color = 'b')
#               plt.ylim(1,-1)
          for col in range(h):
               x, y = xy[:, col, 0], yy[:, col]       
               plt.plot(x,y, color = 'b') 
               plt.ylim(1,-1)
               plt.axis('equal')
     
     elif direct == 1: #Only plot the y-direction 
          for row in range(w):
               x, y = xx[row,:], xy[row,:, 1]       
               plt.plot(x,y, color = 'b')
#               plt.ylim(1,-1)
          for col in range(h):
               x, y = xx[:, col], xy[:, col, 1]       
               plt.plot(x,y, color = 'b') 
               plt.ylim(1,-1)
               plt.axis('equal')
     else:
          for row in range(w):
               x, y = xy[row,:, 0], xy[row,:, 1]       
               plt.plot(x,y, color = 'b')
          for col in range(h):
               x, y = xy[:, col, 0], xy[:, col, 1]       
               plt.plot(x,y, color = 'b') 
               plt.ylim(1,-1)
               plt.axis('equal')


def vis_grid_3d(disp): # xy is of shape h*w*2
     
     w, h, d= np.shape(disp)[0], np.shape(disp)[1], np.shape(disp)[2]
     
     x = np.linspace(-1., 1., w)
     y = np.linspace(-1., 1., h)
     z = np.linspace(-1., 1., d)
     
     xx, yy, zz = np.meshgrid(x, y, z)
     
     xyz = np.stack([xx, yy, zz], 3) + disp
     
     fig = plt.figure()
     ax = fig.add_subplot(111, projection='3d')
     
     for row in range(w):
          for col in range(h):
               x, y, z = xyz[row, col, :, 0], xyz[row, col, :, 1], xyz[row, col, :, 2]
               ax.plot(x,y,z,color = 'b')
     
     for row in range(h):
          for col in range(d):
               x, y, z = xyz[:,row, col,  0], xyz[:,row, col, 1], xyz[:, row, col,  2]
               ax.plot(x,y,z,color = 'b')     

     for row in range(w):
          for col in range(d):
               x, y, z = xyz[row, :, col,  0], xyz[row, :,col,  1], xyz[row, :, col,  2]
               ax.plot(x,y,z,color = 'b')


               

# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 10:08:36 2018

@author: Dongyang

This script contains some utilize functions for data visualization
"""
import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import resize

#==============================================================================
# Iterating Each Slice
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
    print(ax.index)  # could create a slider for it
    
'''
Grid Visualization
'''
#==============================================================================
# 2d grid visualization
# disp: displacement field
# res: ratio of the output size. eg. 0.5 means visualizing with a coarser grid with half the size along each direction
# direct: which direction to show. 0: x, 1:y, 2: both
#==============================================================================
def vis_grid(disp, res = 1, direct = 2): # xy is of shape h*w*2
     
     w, h= np.shape(disp)[0], np.shape(disp)[1]
     
     x = np.linspace(-1., 1., w)
     y = np.linspace(-1., 1., h)
     
     xx , yy = np.meshgrid(x, y)
     
     xy = np.stack([xx,yy], 2) + disp
     
     if res != 1:
          w = np.floor(w*res).astype(int)
          h = np.floor(h*res).astype(int)
     
          dispx = resize(xy[...,0], (w,h), mode='constant', clip = False, order = 1)
          dispy = resize(xy[...,1], (w,h), mode='constant', clip = False, order = 1)
         
          xy = np.stack([dispx, dispy], 2)
     
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

#==============================================================================
# 3d grid visualization
#==============================================================================               
def vis_grid_3d(disp, res = 1):
     
     w, h, d= np.shape(disp)[0], np.shape(disp)[1], np.shape(disp)[2]
         
     x = np.linspace(-1., 1., w)
     y = np.linspace(-1., 1., h)
     z = np.linspace(-1., 1., d)
     
     xx, yy, zz = np.meshgrid(x, y, z)
     
     xyz = np.stack([xx, yy, zz], 3) + disp
     
     if res != 1:
          w = np.floor(w*res).astype(int)
          h = np.floor(h*res).astype(int)
          d = np.floor(d*res).astype(int)
          
          dispx = resize(xyz[...,0], (w,h,d), mode='constant', clip = False, order = 3)
          dispy = resize(xyz[...,1], (w,h,d), mode='constant', clip = False, order = 3)
          dispz = resize(xyz[...,2], (w,h,d), mode='constant', clip = False, order = 3)
          
          xyz = np.stack([dispx, dispy, dispz], 3)
          
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
    

if __name__ == '__main__':
     import SimpleITK as sitk
#     Deli_datapath = r'C:\Users\Dongyang\Downloads\LPBA40\delineation_space\S08\S08.delineation.skullstripped.hdr'
#     Deli_datapath = r'datasets/test_brain_vol.hdr'
     Deli_datapath = r'C:\Users\Dongyang\Downloads\LPBA40\delineation_space\S09\S09.delineation.structure.label.hdr'
     img = sitk.ReadImage(Deli_datapath)
     img_data = sitk.GetArrayViewFromImage(img)
     multi_slice_viewer(img_data,0)
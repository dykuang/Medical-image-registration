# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 10:08:36 2018

@author: Dongyang

This script contains some utilize functions for data visualization
"""
import matplotlib.pyplot as plt
import numpy as np

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
    

if __name__ == '__main__':
     import SimpleITK as sitk
#     Deli_datapath = r'C:\Users\Dongyang\Downloads\LPBA40\delineation_space\S08\S08.delineation.skullstripped.hdr'
#     Deli_datapath = r'datasets/test_brain_vol.hdr'
     Deli_datapath = r'C:\Users\Dongyang\Downloads\LPBA40\delineation_space\S09\S09.delineation.structure.label.hdr'
     img = sitk.ReadImage(Deli_datapath)
     img_data = sitk.GetArrayViewFromImage(img)
     multi_slice_viewer(img_data,0)
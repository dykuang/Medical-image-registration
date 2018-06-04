# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 16:33:45 2018

@author: Dongyang

This script extract slices from .hdr/.img file from LONI LPBA40 dataset
"""

import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt


# =============================================================================
# datapath = r'C:\Users\Dongyang\Downloads\LPBA40\native_space\S09\S09.native.mri.hdr'
# maskdata = r'C:\Users\Dongyang\Downloads\LPBA40\native_space\S09\S09.native.brain.mask.hdr'
# 
# 
# img = sitk.ReadImage(datapath)
# img_data = sitk.GetArrayViewFromImage(img)
# mask = sitk.ReadImage(maskdata)
# mask_data = sitk.GetArrayViewFromImage(mask)
# 
# 
# img_slice = img_data[128,:,:]
# img_slice = img_slice/np.max(img_slice)
# mask_slice = mask_data[128,:,:]/255
# striped = img_slice*mask_slice
# 
# plt.figure()
# plt.subplot(121)
# plt.imshow(img_slice)
# plt.subplot(122)
# plt.imshow(mask_slice)
# 
# plt.figure()
# plt.imshow(striped)
# 
# 
# plt.imsave('slice9.png', striped)
# =============================================================================

# =============================================================================
# from mpl_toolkits.mplot3d import Axes3D
# 
# 
# def midpoints(x):
#     sl = ()
#     for i in range(x.ndim):
#         x = (x[sl + np.index_exp[:-1]] + x[sl + np.index_exp[1:]]) / 2.0
#         sl += np.index_exp[:]
#     return x
# 
# # prepare some coordinates, and attach rgb values to each
# r, g, b = np.indices((17, 17, 17)) / 16.0
# rc = midpoints(r)
# gc = midpoints(g)
# bc = midpoints(b)
# 
# # define a sphere about [0.5, 0.5, 0.5]
# sphere = (rc - 0.5)**2 + (gc - 0.5)**2 + (bc - 0.5)**2 < 0.5**2
# 
# # combine the color components
# colors = np.zeros(sphere.shape + (3,))
# colors[..., 0] = rc
# colors[..., 1] = gc
# colors[..., 2] = bc
# 
# # and plot everything
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax.voxels(r, g, b, sphere,
#           facecolors=colors,
#           edgecolors=np.clip(2*colors - 0.5, 0, 1),  # brighter
#           linewidth=0.5)
# ax.set(xlabel='r', ylabel='g', zlabel='b')
# 
# plt.show()
# =============================================================================


#Deli_datapath = r'C:\Users\Dongyang\Downloads\LPBA40\delineation_space\S09\S09.delineation.structure.label.hdr'
Deli_datapath = r'C:\Users\Dongyang\Downloads\LPBA40\delineation_space\S09\S09.delineation.skullstripped.hdr'
#Deli_datapath = r'datasets/test_brain_vol.hdr'
img = sitk.ReadImage(Deli_datapath)
img_data = sitk.GetArrayViewFromImage(img)
print(img_data.shape)

#img_slice = img_data[90,:,:]
#plt.imsave('slice9.png', img_slice)

from mpl_toolkits.mplot3d import Axes3D
from skimage.transform import resize
fig = plt.figure()
ax = fig.gca(projection='3d')
#ax = Axes3D(fig)
subsample = resize(img_data/np.max(img_data), [16,16,16]) # could cause problems
print('Range of intensity:{}~{}'.format(np.min(subsample), np.max(subsample)))
if np.max(subsample)>1:
     subsample = subsample/np.max(subsample)
color = np.stack((subsample, subsample, subsample, subsample**0.25), axis=3)
ax.voxels(subsample,facecolors = color, edgecolor='k', linestyle='--',
          linewidth=0.1) #facecolor

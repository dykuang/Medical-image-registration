# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 10:08:36 2018

@author: Dongyang

This script contains some utilize functions for data visualization
"""
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np


#==============================================================================
# Define a custom colormap for visualiza Jacobian
#==============================================================================
class MidpointNormalize(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))

#==============================================================================
# Iterating Each Slice
# Modified from 
# datacamp: https://www.datacamp.com/community/tutorials/matplotlib-3d-volumetric-data
#==============================================================================
def remove_keymap_conflicts(new_keys_set):
    for prop in plt.rcParams:
        if prop.startswith('keymap.'):
            keys = plt.rcParams[prop]
            remove_list = set(keys) & new_keys_set
            for key in remove_list:
                keys.remove(key)

def multi_slice_viewer(volume, axis = 0, cmap = 'gray', Jac = False):
    remove_keymap_conflicts({'j', 'k'})
    fig, ax = plt.subplots()

    if axis == 1:
         ax.volume = np.moveaxis(volume, 0, -1)
    elif axis == 2:
         ax.volume = np.moveaxis(volume, [0, 1], [-1, -2])
    else:
         ax.volume = volume
    
    ax.index = volume.shape[0] // 2
    if Jac:
        ax.imshow(ax.volume[ax.index], cmap, norm= MidpointNormalize(midpoint=1))
    else:
        ax.imshow(ax.volume[ax.index], cmap)
        
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


#==============================================================================
#  Overlay two images contained in numpy arrays  
#==============================================================================
def overlay(img1, img2, cmap1=None, cmap2=None, alpha=0.4, Jac = False):
#    plt.figure()
    plt.imshow(img1, cmap=cmap1)
    if Jac:
        plt.imshow(img2, cmap=cmap2, norm=MidpointNormalize(midpoint=1),alpha=alpha)
    plt.imshow(img2, cmap=cmap2, alpha=alpha)
    plt.axis('off')


#==============================================================================
# plot an array of images for comparison
#==============================================================================    
def show_sample_slices(sample_list,name_list, Jac = False, cmap = 'gray', attentionlist=None):
  num = len(sample_list)
  fig, ax = plt.subplots(1,num)
  for i in range(num):
    if Jac:
        ax[i].imshow(sample_list[i], cmap, norm=MidpointNormalize(midpoint=1))
    else:
        ax[i].imshow(sample_list[i], cmap)
    ax[i].set_title(name_list[i])
    ax[i].axis('off')
    if attentionlist:
        ax[i].add_artist(attentionlist[i])
  plt.subplots_adjust(wspace=0)
  
def my_cmap(name='mycmap', colors=[(0, 0, 0), (0, 1, 0), (1, 0, 0)]):
    from matplotlib.colors import LinearSegmentedColormap
    cm = LinearSegmentedColormap.from_list(name, colors, N=len(colors))
    return cm

#def examine_ROI(A, B):
#    '''
#    A, B are masks of ROI to be examined
#    '''
#    plt.figure()
#    plt.imshow(A, cmap=my_cmap(color))
    

'''
Grid Visualization
'''
#==============================================================================
# 2d grid visualization
# disp: displacement field
# res: ratio of the output size. eg. 0.5 means visualizing with a coarser grid with half the size along each direction
# direct: which direction to show. 0: x, 1:y, 2: both
#==============================================================================
from skimage.transform import resize
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
    
#==============================================================================
# Generating random colors, copied from https://github.com/delestro/rand_cmap
#==============================================================================
def rand_cmap(nlabels, type='bright', first_color_black=True, last_color_black=False, verbose=True):
    """
    Creates a random colormap to be used together with matplotlib. Useful for segmentation tasks
    :param nlabels: Number of labels (size of colormap)
    :param type: 'bright' for strong colors, 'soft' for pastel colors
    :param first_color_black: Option to use first color as black, True or False
    :param last_color_black: Option to use last color as black, True or False
    :param verbose: Prints the number of labels and shows the colormap. True or False
    :return: colormap for matplotlib
    """
    from matplotlib.colors import LinearSegmentedColormap
    import colorsys
    import numpy as np


    if type not in ('bright', 'soft'):
        print ('Please choose "bright" or "soft" for type')
        return

    if verbose:
        print('Number of labels: ' + str(nlabels))

    # Generate color map for bright colors, based on hsv
    if type == 'bright':
        randHSVcolors = [(np.random.uniform(low=0.0, high=1),
                          np.random.uniform(low=0.2, high=1),
                          np.random.uniform(low=0.9, high=1)) for i in range(nlabels)]

        # Convert HSV list to RGB
        randRGBcolors = []
        for HSVcolor in randHSVcolors:
            randRGBcolors.append(colorsys.hsv_to_rgb(HSVcolor[0], HSVcolor[1], HSVcolor[2]))

        if first_color_black:
            randRGBcolors[0] = [0, 0, 0]

        if last_color_black:
            randRGBcolors[-1] = [0, 0, 0]

        random_colormap = LinearSegmentedColormap.from_list('new_map', randRGBcolors, N=nlabels)

    # Generate soft pastel colors, by limiting the RGB spectrum
    if type == 'soft':
        low = 0.6
        high = 0.95
        randRGBcolors = [(np.random.uniform(low=low, high=high),
                          np.random.uniform(low=low, high=high),
                          np.random.uniform(low=low, high=high)) for i in range(nlabels)]

        if first_color_black:
            randRGBcolors[0] = [0, 0, 0]

        if last_color_black:
            randRGBcolors[-1] = [0, 0, 0]
        random_colormap = LinearSegmentedColormap.from_list('new_map', randRGBcolors, N=nlabels)

    # Display colorbar
    if verbose:
        from matplotlib import colors, colorbar
        from matplotlib import pyplot as plt
        fig, ax = plt.subplots(1, 1, figsize=(15, 0.5))

        bounds = np.linspace(0, nlabels, nlabels + 1)
        norm = colors.BoundaryNorm(bounds, nlabels)

        cb = colorbar.ColorbarBase(ax, cmap=random_colormap, norm=norm, spacing='proportional', ticks=None,
                                   boundaries=bounds, format='%1i', orientation=u'horizontal')

    return random_colormap

  
#if __name__ == '__main__':
#    import SimpleITK as sitk
#
#    indmb = 72
#    f8081 = np.load('080081incept0.npy')
#    v8081 = np.load('080081V.npy')[...,0]
#    S80 = np.load('080.npy')
#    S81 = np.load('081.npy')
#    u8081 = np.load('u080081.npy')
#    samplemb = [S80[indmb,::-1,:], u8081[indmb,::-1,:], v8081[indmb,::-1,:], f8081[indmb,::-1,:], S81[indmb,::-1,:]]
#    namesmb = ['OASIS-TRT-20-3','UtilzReg','VoxelMorph','FAIM','OASIS-TRT-20-8']
#    
#    fJ = np.load('080081incept0JAC.npy')[0]
#    vJ = np.load('080081JAC.npy')[0]
#    uJ = np.load('u080081JAC.npy')
#    Jmb = [uJ[indmb,::-1,:], vJ[indmb,::-1,:], fJ[indmb,::-1,:]]
#    
#    indlb = 45
#    S32 = np.load('S32.npy')
#    S39 = np.load('S39.npy')
#    f3239 = np.load('3239incept0.npy')
#    v3239 = np.load('S3239V.npy')
#    udat = sitk.ReadImage('S3239.nii')
#    u3239 = sitk.GetArrayFromImage(udat)
#    u3239 = u3239/np.max(u3239)
#    nameslb=['S32','UtilzReg','VoxelMorph','FAIM','S39']
#    samplelb = [S32[indlb,::-1,:], u3239[indlb,::-1,:], v3239[indlb,::-1,:], f3239[indlb,::-1,:], S39[indlb,::-1,:]]
#    
#    fJlb = np.load('3239incept0JAC.npy')[0]
#    vJlb = np.load('S3239Vflow.npy')[0]
#    uJlb = np.load('S3239uflow.npy')[0]
#    Jlb = [uJlb[indlb,::-1,:], vJlb[indlb,::-1,:], fJlb[indlb,::-1,:]]
#    
#    
#    circlist = [plt.Circle((98,125), 25, color='r',fill=False) for i in range(5)]
#    show_sample_slices(samplemb, namesmb, attentionlist=circlist)
#    circlist2 = [plt.Circle((98,125), 25, color='k',fill=False) for i in range(3)]
#    show_sample_slices(Jmb, namesmb[1:-1], Jac=True, cmap='bwr_r', attentionlist=circlist2)
#    
#    show_sample_slices(samplelb, nameslb)
#    show_sample_slices(Jlb, nameslb[1:-1], Jac=True, cmap='bwr_r')
#    
#    
#    ve3 = np.load('lpba3239incept0VJAC_tvse-3.npy')[0]
#    v001 = np.load('lpba3239incept0VJAC_tvs001.npy')[0]
#    v01 = np.load('lpba3239incept0VJAC_tvs01.npy')[0]
##    v05 = np.load('lpba3239incept0VJAC_tvs05.npy')[0]
#    v1 = np.load('lpba3239incept0VJAC_tvs1.npy')[0]
#    
#    vtvs=[vJlb[indlb,::-1,:], ve3[indlb,::-1,:],v001[indlb,::-1,:], v01[indlb,::-1,:], v1[indlb,::-1,:]]
#    namesv=['0', '0.001', '0.01', '0.1', '1']
#    show_sample_slices(vtvs,namesv,Jac=True, cmap = 'bwr_r')
#    
#    f001 = np.load('3239incept0JACtvs001.npy')[0]
#    f01 = np.load('3239incept0JACtvs01.npy')[0]
#    f1 = np.load('3239incept0JACtvs1.npy')[0]
#    f1e3 = np.load('3239incept0JACtvs1e-3.npy')[0]
#    ftvs=[fJlb[indlb,::-1,:], f1e3[indlb,::-1,:],f001[indlb,::-1,:], f01[indlb,::-1,:], f1[indlb,::-1,:]]
#    show_sample_slices(ftvs,namesv,Jac=True, cmap = 'bwr_r')
#    
#    
#    fJR2 = np.load('3239incept0JACR2.npy')[0]
#    fJR4 = np.load('3239incept0JACR4.npy')[0]
#    fJR05 = np.load('3239incept0JACR05.npy')[0]
#    fJR8 = np.load('3239incept0JACR8.npy')[0]
#    namesR = ['0.5', '1', '2', '4', '8']
#    fJR=[fJR05[indlb,::-1,:],fJlb[indlb,::-1,:], fJR2[indlb,::-1,:],fJR4[indlb,::-1,:], fJR8[indlb,::-1,:]]
#    show_sample_slices(fJR,namesR,Jac=True, cmap = 'bwr_r')
#    
#    vJR2 = np.load('lpba3239incept0VJAC_R2.npy')[0]
#    vJR4 = np.load('lpba3239incept0VJAC_R4.npy')[0]
#    vJR05 = np.load('lpba3239incept0VJAC_R05.npy')[0]
#    vJR8 = np.load('lpba3239incept0VJAC_R8.npy')[0]
#
#    vJR=[vJR05[indlb,::-1,:],vJlb[indlb,::-1,:], vJR2[indlb,::-1,:],vJR4[indlb,::-1,:], vJR8[indlb,::-1,:]]
#    show_sample_slices(vJR,namesR,Jac=True, cmap = 'bwr_r')
#    
#     
#    '''
#    lpba atlas
#    '''
#    lpbavg = np.load('lpbavg.npy')
#    lpbaprob = np.load('lpba_maxprob.npy')
#    lpbalabel = np.load('lpbavglabel.npy')
#    
#    favg = np.load('lpba_avg_lr3.npy')
#    fprob = np.load('lpba_maxprob_lr3.npy')
#    flabel = np.load('lpba_WinningLabel_lr3.npy')
#    
#    show_sample_slices([lpbavg[45,::-1,:], favg[45,::-1,:]], name_list=[' ', ' '])
#    show_sample_slices([lpbaprob[55,::-1,:], fprob[55,::-1,:]], name_list=[' ', ' '])
#    plt.figure()
#    plt.subplot(1,2,1)
#    overlay(  lpbavg[55,::-1,:], lpbaprob[55,::-1,:],'gray','jet', alpha = 0.6)
#    plt.subplot(1,2,2)
#    overlay(  favg[55,::-1,:],fprob[55,::-1,:], 'gray','jet', alpha = 0.6)
#    plt.subplots_adjust(hspace=0, wspace=0)
#    
#    plt.figure()
#    plt.subplot(1,2,1)
#    overlay(lpbavg[::-1,55,:],lpbaprob[::-1,55,:],'gray','jet', alpha = 0.6)
#    plt.subplot(1,2,2)
#    overlay(favg[::-1,55,:], fprob[::-1,55,:],'gray', 'jet', alpha = 0.6)
#    plt.subplots_adjust(hspace=0, wspace=0)
#    
#    plt.figure()
#    plt.subplot(1,2,1)
#    overlay(lpbavg[::-1,:,55], lpbaprob[::-1,:,55], 'gray','jet', alpha = 0.6)
#    plt.subplot(1,2,2)
#    overlay(favg[::-1,:,55], fprob[::-1,:,55], 'gray','jet', alpha = 0.6)
#    plt.subplots_adjust(hspace=0, wspace=0)
#    
#    
#    '''
#    MindBoggle atlas
#    '''
#    mbavg = np.load('Otemp.npy')
#    mbprob = np.load('Oprob.npy')
#    mblabel = np.load('OtempLabel.npy')
#    mbprob = mbprob*(mblabel>1000) # mask out non cortex region
#    mblabel = mblabel*(mblabel>1000)
#    
#    mbfavg = np.load('mb_avg_incept0_temp1.npy')
#    mbfprob = np.load('mb_maxprob_temp1.npy')
#    mbflabel = np.load('mb_WinningLabel_temp1.npy') 
#    mbfprob = mbfprob*(mbflabel>0) #mask out the background
#    
#    show_sample_slices([mbavg[72,...], mbfavg[72,...]], name_list=[' ', ' '])
##    show_sample_slices([lpbalabel[45,...], mblabel[72,...]], name_list=['LPBA40', 'MindBoggle'])
#    plt.figure()
#    plt.subplot(1,2,1)
#    overlay(mbavg[72,::-1,:], mbprob[72,::-1,:], 'gray','jet', alpha = 0.6)
#    plt.subplot(1,2,2)
#    overlay(mbavg[72,::-1,:], mbfprob[72,::-1,:], 'gray','jet', alpha = 0.6)   
#    plt.subplots_adjust(hspace=0, wspace=0)
#    
#    plt.figure()
#    plt.subplot(1,2,1)
#    overlay(mbavg[::-1,94,:], mbprob[::-1,94,:], 'gray','jet', alpha = 0.6)
#    plt.subplot(1,2,2)
#    overlay(mbavg[::-1,94,:], mbfprob[::-1,94,:], 'gray','jet', alpha = 0.6)
#    plt.subplots_adjust(hspace=0, wspace=0)
#    
#    plt.figure()
#    plt.subplot(1,2,1)
#    overlay(mbavg[::-1,:, 74], mbprob[::-1,:,74], 'gray','jet', alpha = 0.6)
#    plt.subplot(1,2,2)
#    overlay(mbavg[::-1,:, 74], mbfprob[::-1,:,74], 'gray','jet', alpha = 0.6)
#    plt.subplots_adjust(hspace=0, wspace=0)
#    
#    
#    label_list = np.unique(mblabel*(mblabel>1000))
#    fig, ax = plt.subplots(1,3)
#    MBroi47 = (mblabel==label_list[47])[72,110:135,10:40]
#    Froi47 = (mbflabel==[47])[72,110:135,10:40]
#    ax[0].imshow(MBroi47, my_cmap(colors=[(1,1,1),(0,1,0)]))
#    ax[0].imshow(Froi47, my_cmap(colors=[(1,1,1),(0,0,1)]), alpha = 0.5)
#    ax[0].set_axis_off()
#    ax[0].set_title('parso percularis')
#    
#    MBroi56 = (mblabel==label_list[56])[72,142:,20:74]
#    Froi56 = (mbflabel==[56])[72,142:,20:74]
#    ax[1].imshow(MBroi56, my_cmap(colors=[(1,1,1),(0,1,0)]))
#    ax[1].imshow(Froi56, my_cmap(colors=[(1,1,1),(0,0,1)]), alpha = 0.5)
#    ax[1].axis('off')
#    ax[1].set_title('rostral middlefrontal')
#    
#    MBroi59 = (mblabel==label_list[59])[72,65:98,:32]
#    Froi59 = (mbflabel==[59])[72,65:98,:32]
#    ax[2].imshow(MBroi59, my_cmap(colors=[(1,1,1),(0,1,0)]))
#    ax[2].imshow(Froi59, my_cmap(colors=[(1,1,1),(0,0,1)]), alpha = 0.5)
#    ax[2].axis('off')
#    ax[2].set_title('superior temporal')
#    plt.subplots_adjust(hspace=0, wspace=0)
    
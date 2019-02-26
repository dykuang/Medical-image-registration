#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 17 09:53:07 2018

@author: dykuang

plot the comparison for dice score
"""

import numpy as np
import matplotlib.pyplot as plt


def Dice_group(dice, grouped_ind):
    '''
    dice should be a 2d array
    the function returns the mean dice for each group of ROI
    median or mean?
    '''
    group_dice = [np.mean(dice[:, ind],1) for ind in grouped_ind]
  
    return group_dice


def Dice_bar(Before, After_sdn, After_utr, labellist):
    # Setting the positions and width for the bars
    pos = list(range(len(labellist))) 
    width = 0.2 
        
    # Plotting the bars
    fig, ax = plt.subplots(figsize=(8,30))
    
    # Create a bar with pre_score data,
    # in position pos,
    plt.barh(pos, 
            #using df['pre_score'] data,
            Before, 
            # of width
            width, 
            # with alpha 0.5
            alpha=0.5, 
            # with color
            color='g', 
            # with label the first value in first_name
            label= ['before']) 
    
    # Create a bar with mid_score data,
    # in position pos + some width buffer,
    plt.barh([p + width for p in pos], 
            #using df['mid_score'] data,
            After_sdn,
            # of width
            width, 
            # with alpha 0.5
            alpha=0.8, 
            # with color
            color= 'k', 
            # with label the second value in first_name
            label=['after_sdn']) 
    
    # Create a bar with post_score data,
    # in position pos + some width buffer,
    plt.barh([p + width*2 for p in pos], 
            #using df['post_score'] data,
            After_utr, 
            # of width
            width, 
            # with alpha 0.5
            alpha=0.5, 
            # with color
            color= 'r', 
            # with label the third value in first_name
            label=['after_utr']) 
    
    # Set the y axis label
    ax.set_xlabel('Dice')
    ax.set_ylabel('ROI label')
    # Set the chart's title
    ax.set_title('ROI Dices')
    
    # Set the position of the x ticks
    ax.set_yticks([p + 1.0 * width for p in pos])
    
    # Set the labels for the x ticks
    ax.set_yticklabels(labellist)
    
    # Setting the x-axis and y-axis limits
    plt.ylim(min(pos)-width, max(pos)+width*4)
    plt.xlim([0, 1.0])
    plt. yticks(rotation = 0)
    # Adding the legend and showing the plot
    plt.legend(['Before', 'After_sdn', 'After_utr'], loc='upper left')
    plt.grid()
    plt.show()
    

def Dice_box(data, ticklabel = ['Before', 'After_sdn', 'After_utr']): # data is a list of 1d array
    fig = plt.figure(figsize=(9,6))
    ax = fig.add_subplot(111)
    ## to get fill color
    bp = ax.boxplot(data, patch_artist=True, widths = .2)
    
    ## change outline color, fill color and linewidth of the boxes
    for box in bp['boxes']:
        # change outline color
        box.set( color='#7570b3', linewidth=2)
        # change fill color
        box.set( facecolor = '#1b9e77')
    
    ## change color and linewidth of the whiskers
    for whisker in bp['whiskers']:
        whisker.set(color='#7570b3', linewidth=2)
    
    ## change color and linewidth of the caps
    for cap in bp['caps']:
        cap.set(color='#7570b3', linewidth=2)
    
    ## change color and linewidth of the medians
    for median in bp['medians']:
        median.set(color='#b2df8a', linewidth=2)
    
    ## change the style of fliers and their fill
    for flier in bp['fliers']:
        flier.set(marker='o', color='#e7298a', alpha=0.5)
                  
    ax.set_xticklabels(ticklabel)
    ## Remove top axes and right axes ticks
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.axhline(np.percentile(data[1], 25), linestyle = '-.', color = 'r')
    ax.axhline(np.percentile(data[1], 75), linestyle = '-.', color = 'r')
    plt.ylim([0.5, 0.8])
    plt.ylabel('Dice')
    plt.grid()
    
   
#==============================================================================
# A grouped boxplot
#==============================================================================

def set_box_color(bp, color):
    plt.setp(bp['boxes'], color = color, facecolor = color, linewidth=2)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
#    plt.setp(bp['medians'], color=color)
    plt.setp(bp['medians'], linewidth=2)
    plt.setp(bp['fliers'], markersize=4)

def show_dice_group(data, names, ticks, colors, box_width = 0.3, sparsity = 3, ymin=0, ymax = 1):
    plt.figure()
    for i, sample in enumerate(data):
        bp = plt.boxplot(sample, positions=np.array(np.arange(len(sample)))*sparsity-0.6+0.4*i,  widths=box_width, sym = 'o',
                  notch=True, patch_artist=True)
        set_box_color(bp, colors[i])
        for patch in bp['boxes']:
           patch.set_alpha(0.8)
        plt.plot([], c=colors[i], label=names[i])
    plt.legend(loc='lower left')

    plt.xticks(np.arange(0, len(ticks) * sparsity, sparsity), ticks, rotation = 45)
    plt.xlim(-2, len(ticks)*sparsity-0.4)
    plt.ylim(ymin, ymax)
    plt.ylabel('Dice Score')
    #plt.title('Different methods on selected regions')
    plt.grid()
    plt.tight_layout()
        
#==============================================================================
# A grouped barplot
#==============================================================================
def Dice_bar_group(data, group_names, legend_names, colors):
    '''
    Input:
        data: a list of vectors
        group_names: a list of group names
        legend_names: a list of names for each bar within a group
        colors: a list of color code for each bar within a group
    '''
    # Setting the positions and width for the bars
    pos = list(range(len(group_names))) 
    width = 0.1
        
    # Plotting the bars
    fig, ax = plt.subplots(figsize=(8,30))
    
    # Create a bar with pre_score data,
    # in position pos,
    for i, vec in enumerate(data):
        plt.barh([p + width*i for p in pos], 
                #using df['pre_score'] data,
                vec, 
                # of width
                width, 
                # with alpha 0.5
                alpha=1, 
                # with color
                color=colors[i], 
                # with label the first value in first_name
                label= group_names[i]) 
    
    # Set the y axis label
    ax.set_xlabel('Dice')
    ax.set_ylabel('ROI label')
    # Set the chart's title
    ax.set_title('Mean ROI Dices')
    
    # Set the position of the x ticks
    ax.set_yticks([p + 1.75 * width for p in pos])
    
    # Set the labels for the x ticks
    ax.set_yticklabels(group_names)
    
    # Setting the x-axis and y-axis limits
    plt.ylim(min(pos)-width, max(pos)+width*len(group_names))
    plt.xlim([0, 1.0])
    plt. yticks(rotation = 0)
    # Adding the legend and showing the plot
    plt.legend(legend_names, loc='upper right')
    plt.grid()
    plt.show()
    
#if __name__ == '__main__':
#    '''
#    LPBA40 results
#    '''
#    before = np.load('dice_before_ver6.npy')
#    after_utr6 = np.load('dice_after_utr6.npy')
#    voxdc = np.load('Vox_dice_lpba_dc.npy')
#    voxdclre3 = np.load('Vox_lpba_lr1e-3.npy')
#    
#    lpba_incept=np.load('lpba_dice_incept.npy')
#    lpba_inceptlr3 = np.load('lpba_dice_inceptlr3.npy')
#    lpba_incept0 = np.load('lpba_dice_incept0.npy')
#    lpba_incept0lr3 = np.load('lpba_dice_incept0lr3.npy')
#    lpba_incept4lr3 = np.load('lpba_dice_incept4lr3.npy')
#    
#    
#    label_list = [i for i in range(21,35,1)] + [i for i in range(41,51,1)] + [i for i in range(61, 69, 1)] + [i for i in range(81,93,1)] + [101, 102, 121, 122] + [i for i in range(161, 167, 1)] + [181, 182]
#    FL = [i for i in range(14)]
#    PL = [i for i in range(14, 24)]
#    OL = [i for i in range(24, 32)]
#    TL = [i for i in range(32, 44)]
#    CL = [i for i in range(44, 48)]
#    Ptm = [50, 51]
#    Hpcp = [52, 53]
#    
#    ticks = ['Frontal Lobe', 'Parietal Lobe', 'Occipital Lobe', 'Temporal Lobe', 'Cingulate Lobe', 'Putamen', 'Hippocampus']
#    colors = ['#2C7BB6', '#2ca25f', '#636363', '#D7191C', '#68228B']
#    box_width = 0.3
#    sparsity = 3
#    
#    group_ind = [FL, PL, OL, TL, CL, Ptm, Hpcp]
#    flirt = Dice_group(before,group_ind)
#    incept = Dice_group(lpba_incept0,group_ind)
#    vox = Dice_group(voxdc,group_ind)
#    utr = Dice_group(after_utr6,group_ind)
#    incept_best = Dice_group(lpba_incept0lr3,group_ind)
#    names=['Before', 'UtilzReg', 'VoxelMorph', 'FAIM', 'FAIM(lr=1e-3)']
#    data = [flirt, utr, vox, incept, incept_best]
#    
#    show_dice_group(data, names, ticks, colors, ymin=0.2, ymax=0.8)
#    
#    btemp = np.load('lpba_dice2temp_before.npy')
#    V2temp = np.load('lpba_dice2tempV.npy')
#    F2temp = np.load('lpba_dice2temp.npy')
#    
#    G2temp = [Dice_group(s,group_ind) for s in [btemp, V2temp, F2temp]]
#    show_dice_group(G2temp, ['Before','VoxelMorph', 'FAIM'], ticks, colors=['#2C7BB6','#636363', '#D7191C'], ymin=0.2, ymax=0.8)
#        
#    
#    '''
#    MindBoggle results
#    '''
#    label_list_25 = [1002, 1003, 1005, 1006, 1007, 1008, 1009, 1011,
#       1012, 1013, 1014, 1015, 1016, 1017, 1018, 1021, 1022,
#       1024, 1025, 1028, 1029, 1030, 1031, 1034, 1035, 2002,
#       2003, 2005, 2006, 2007, 2008, 2009, 2011, 2012, 2013,
#       2014, 2015, 2016, 2017, 2018, 2021, 2022, 2024, 2025,
#       2028, 2029, 2030, 2031, 2034, 2035]
#
#    Frontal_label = [1000 + i for i in[3, 12, 14, 17, 18, 19, 20, 24, 27, 28]] + [2000+ j for j in[3, 12, 14, 17, 18, 19, 20, 24, 27, 28]]
#    Parietal_label = [1000 + i for i in [8, 22, 25, 29, 31]] + [2000 + j for j in [8, 22, 25, 29, 31]]
#    Occipital = [1000 + i for i in [5, 11, 13, 21]] + [2000 + j for j in [5, 11, 13, 21]]
#    Temporal = [1000 + i for i in [6, 7, 9, 15, 16, 30, 34]] + [2000 + i for i in [6, 7, 9, 15, 16, 30, 34]]
#    Cingulate = [1000 + i for i in [2, 10, 23, 26, 35]] + [2000 + i for i in [2, 10, 23, 26, 35]]
#    
#    def find_label(group, label_list = label_list_25): 
#    '''
#    Find label index of given group in the label_list
#    '''
#        label_found = []
#        for i in range(len(label_list)):
#            if label_list[i] in group:
#                label_found.append(i)
#        return label_found
#    
#    Frontal = find_label(Frontal_label)
#    Parietal = find_label(Parietal_label)
#    Occipital = find_label(Occipital)
#    Temporal = find_label(Temporal)
#    Cingulate= find_label(Cingulate)
#    
#    Dice_before = np.load('dice_before.npy')
#    Dice_UTR = np.load('Dice_after_UTR20.npy')
#    vdc = np.load('Vox_dice_dc2.npy')
#    voxlr5e4 = np.load('Vox_dice_lr5e-4.npy')
#    dice17_4 = np.load('dice_after_TVS17_4.npy')
#    dice17_5 = np.load('dice_after_17_5.npy')
#    diceincept = np.load('dice_after_incept.npy')
#    diceincept2 = np.load('dice_after_incept2.npy')
#    diceincept3 = np.load('dice_after_incept3.npy')
#    diceincept4 = np.load('dice_after_incept4.npy')
#    diceinceptG = np.load('dice_after_inceptG.npy')
#    diceincept0 = np.load('dice_after_incept0.npy')
#    diceincept0lr5e4 = np.load('dice_after_incept0lr5e-4.npy')
#    
#    ROI_ind = [Frontal, Parietal, Occipital, Temporal, Cingulate]
#    Before = Dice_group(Dice_before,ROI_ind)
#    Faim = Dice_group(diceincept0,ROI_ind)
#    vox_mb = Dice_group(vdc,ROI_ind)
#    utr_mb = Dice_group(Dice_UTR,ROI_ind)
#    Faim_best = Dice_group(diceincept0lr5e4,ROI_ind)
#    
#    names_mb=['Before', 'UtilzReg', 'VoxelMorph', 'FAIM', 'FAIM(lr=5e-4)']
#    data_mb = [Before, utr_mb, vox_mb, Faim, Faim_best]
#    show_dice_group(data_mb, names_mb, ticks[:-2], colors, ymin=0.0, ymax=0.8)
#    
#    
#    btemp_mb = np.load('mb_dice2temp_before.npy')
#    V2temp_mb = np.load('mb_dice2tempV.npy')
#    F2temp_mb = np.load('mb_dice2temp.npy')
#    
#    G2temp_mb = [Dice_group(s, ROI_ind) for s in [btemp_mb, V2temp_mb, F2temp_mb]]
#    show_dice_group(G2temp_mb, ['Before','VoxelMorph', 'FAIM'], ticks[:-2], colors=['#2C7BB6','#636363', '#D7191C'], ymin=0.0, ymax=0.8)
    
    
    
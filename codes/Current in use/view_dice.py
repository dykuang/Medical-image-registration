#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 17 09:53:07 2018

@author: dykuang

plot the comparison for dice score
"""

import numpy as np
import matplotlib.pyplot as plt

before = np.load('dice_before_ver6.npy')
after_utr6 = np.load('dice_after_utr6.npy')
#after_sdn = np.load('dice_after_ver6.npy')
#after_sdn_0 = np.load('dice_after_ver6_5.npy')
##after_sdn_1 = np.load('dice_after_ver6_a1e-4.npy')
##after_sdn_2 = np.load('dice_after_ver6_r1e-5.npy')
#after_sdn_3 = np.load('dice_after_ver6_r2e-5.npy')
##after_sdn_4 = np.load('dice_after_ver6_5r1e-5.npy')
##after_sdn_5 = np.load('dice_after_ver6_r1e-6.npy')
##after_sdn_6 = np.load('dice_after_ver6_5a2e-3.npy')
##after_sdn_7 = np.load('dice_after_ver7.npy')
##after_sdn_8 = np.load('dice_after_ver6_5r2e-5.npy')
##after_sdn_9 = np.load('dice_after_5r5e-5.npy')
##after_sdn_10 = np.load('dice_after_a5e-4r2e-5.npy')
##after_sdn_11 = np.load('dice_after_2_0.npy')
##after_sdn_12 = np.load('dice_after_0_1.npy')
##after_sdn_13 = np.load('dice_after_1_2.npy')
#after_sdn_14 = np.load('dice_after_branched.npy')
#after_sdn_15 = np.load('dice_after_whole.npy')
#after_sdn_17 = np.load('dice_after_1_1.npy')
#after_sdn_18 = np.load('dice_after_3_1.npy')
#after_sdn_19 = np.load('dice_after_4_1.npy')
#after_sdn_20 = np.load('dice_after_2112.npy')
#
#after_sdn_8_1 = np.load('dice_after_8_1.npy')
#after_sdn_Pr3 = np.load('dice_after_Pr3.npy')
#after_sdn_143= np.load('dice_after_143.npy')
#after_sdn_148 = np.load('dice_after_148.npy')
#
#after_sdn_2pass = np.load('dice_after_2pass.npy')
#
#after_sdn_158 = np.load('dice_after_158.npy')
#
#
#after_sdn_169 = np.load('dice_after_169.npy')
#after_sdn_172 = np.load('dice_after_172.npy')
#after_sdn_173 = np.load('dice_after_173.npy')
#
#
#after_sdn_174 = np.load('dice_after_174.npy')
#after_sdn_175 = np.load('dice_after_175.npy')
#after_sdn_176 = np.load('dice_after_176.npy')
#
#after_sdn_177 = np.load('dice_after_177.npy')
#after_sdn_178 = np.load('dice_after_178.npy')
#after_sdn_186 = np.load('dice_after_PR.npy')
#after_sdn_188 = np.load('dice_after_188.npy')
#after_sdn_189 = np.load('dice_after_189.npy')
#
#after_sdn_192 = np.load('dice_after_192.npy')
#after_sdn_198 = np.load('dice_after_198.npy')
#after_sdn_205 = np.load('dice_after_205.npy')
#after_sdn_207 = np.load('dice_after_207.npy')
#
#after_sdn_209 = np.load('dice_after_209.npy')
#after_sdn_210 = np.load('dice_after_210.npy')
#
#
#after_sdn_211 = np.load('dice_after_211.npy')
#after_sdn_212 = np.load('dice_after_212.npy')
#after_sdn_213 = np.load('dice_after_213.npy')
#after_sdn_214 = np.load('dice_after_214.npy')
#
#after_sdn_215 = np.load('dice_after_215.npy')
#after_sdn_223 = np.load('dice_after_223.npy')
#
#after_sdn_224 = np.load('dice_after_224.npy')
#after_sdn_228 = np.load('dice_after_228.npy')
#
#after_sdn_234 = np.load('dice_after_234.npy')
#
#after_sdn_236 = np.load('dice_after_236.npy')
#after_sdn_236DC = np.load('dice_after_236_dc.npy')
#after_sdn_236tanh = np.load('dice_after_236_tanh.npy')
#after_sdn_238 = np.load('dice_after_238.npy')
#after_sdn_2381 = np.load('dice_after_238_1.npy')
#
#after_sdn_240 = np.load('dice_after_240.npy')
after_sdn_241 = np.load('dice_after_241.npy')
after_sdn_243 = np.load('dice_after_243.npy')
#after_sdn_243CD = np.load('dice_after_243_CD.npy') # check Dice score implementation
#after_sdn_244 = np.load('dice_after_244.npy')
#after_sdn_253 = np.load('dice_after_253.npy')
#after_sdn_253DC = np.load('dice_after_253_DC.npy') # double check performance with interpnn
#after_sdn_256 = np.load('dice_after_256.npy')
#after_sdn_257 = np.load('dice_after_257.npy')
#after_sdn_258 = np.load('dice_after_258.npy')
#
#after_sdn_290 = np.load('dice_after_290.npy')
#after_sdn_291 = np.load('dice_after_291.npy')
#after_sdn_292 = np.load('dice_after_292.npy')
#
Vox_dice_16 = np.load('Vox_dice_16.npy')
#Vox_dice_16_1 = np.load('Vox_dice_16_1.npy')
#
#after_utr = np.load('dice_after_utr.npy')
#
#before_old_test = np.load('dice_before_testlist.npy')
#after_old_test = np.load('dice_after_sdn1.npy')
#
#before_de = np.load('dice_before_de.npy')
#after_de = np.load('dice_after_3_1_de.npy')
#
#before_ctx_158 = np.load('dice_before_ctx_158.npy')
#after_ctx_158 = np.load('dice_after_ctx_158.npy')
#plt.figure()
#plt.plot(before[0])
#plt.plot(after_utr[0])
#plt.plot(after_sdn[0])
#plt.legend(['Before', 'Utr', 'SDN'])
#plt.ylim([0.3, 0.9])


after_sdn_304 = np.load('dice_after_304.npy')
after_sdn_305 = np.load('dice_after_305.npy')
after_sdn_309 = np.load('dice_after_309.npy')

label_list = [i for i in range(21,35,1)] + [i for i in range(41,51,1)] + [i for i in range(61, 69, 1)] + [i for i in range(81,93,1)] + [101, 102, 121, 122] + [i for i in range(161, 167, 1)] + [181, 182]


def Dice_group(dice):
    '''
    dice should be a 2d array
    the function returns the mean dice for each group of ROI
    median or mean?
    '''
    
    group_dice = [np.mean(dice[:,:6],1), 
                  np.mean(dice[:,24:30],1),
                  np.mean(dice[:,32:38],1),
                  np.mean(dice[:,50:52],1),
                  np.mean(dice[:,52:54],1),
                  np.mean(dice[:,54:55],1)]
    
    return group_dice
    


def Dice_bar(Before, After_sdn, After_utr, labellist=label_list):
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
    ax.axhline(np.percentile(data[3], 25), linestyle = '-.', color = 'r')
    ax.axhline(np.percentile(data[3], 75), linestyle = '-.', color = 'r')
    plt.ylim([0.5, 0.8])
    plt.ylabel('Dice')
    plt.grid()
    
#==============================================================================
# some scratches below
#==============================================================================
# =============================================================================
# 
    
#==============================================================================
# A grouped boxplot
#==============================================================================
ticks = ['Frontal gyrus', 'Occipital gyrus', 'Temporal gyrus', 'Putamen', 'Hippocampus', 'Cerebellum']
colors = ['#2C7BB6', '#D7191C', '#2ca25f', '#636363']
box_width = 0.3
sparsity = 3

def set_box_color(bp, color):
    plt.setp(bp['boxes'], color = color, facecolor = color, linewidth=2)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
#    plt.setp(bp['medians'], color=color)
    plt.setp(bp['medians'], linewidth=2)
    plt.setp(bp['fliers'], markersize=4)

flirt = Dice_group(before)
incept = Dice_group(after_sdn_243)
vox = Dice_group(Vox_dice_16)
utr = Dice_group(after_utr6)

#flirt = [np.random.random_sample(20) for _ in range(6)]
#incept = [np.random.random_sample(20) for _ in range(6)]
#vox = [np.random.random_sample(20) for _ in range(6)]
#utr = [np.random.random_sample(20) for _ in range(6)]

bp1 = plt.boxplot(flirt, positions=np.array(np.arange(len(flirt)))*sparsity-0.6,  widths=box_width, sym = 'o',
                  notch=True, patch_artist=True)
bp2 = plt.boxplot(incept, positions=np.array(np.arange(len(incept)))*sparsity-0.2, widths=box_width, sym = 'o',
                  notch=True, patch_artist=True)
bp3 = plt.boxplot(vox, positions=np.array(np.arange(len(vox)))*sparsity+0.2, widths=box_width, sym = 'o',
                  notch=True, patch_artist=True)
bp4 = plt.boxplot(utr, positions=np.array(np.arange(len(utr)))*sparsity+0.6, widths=box_width, sym = 'o',
                  notch=True, patch_artist=True)

set_box_color(bp1, colors[0]) # colors are from http://colorbrewer2.org/
set_box_color(bp2, colors[1])
set_box_color(bp3, colors[2]) 
set_box_color(bp4, colors[3])

for patch in bp1['boxes']:
    patch.set_alpha(0.8)
for patch in bp2['boxes']:
    patch.set_alpha(0.8)
for patch in bp3['boxes']:
    patch.set_alpha(0.8)
for patch in bp4['boxes']:
    patch.set_alpha(0.8)
              
# draw temporary red and blue lines and use them to create a legend
plt.plot([], c=colors[0], label='Before')
plt.plot([], c=colors[1], label='1 inception')
plt.plot([], c=colors[2], label='VoxelMorph-1')
plt.plot([], c=colors[3], label='UtilzReg')
plt.legend(loc='lower right')

plt.xticks(np.arange(0, len(ticks) * sparsity, sparsity), ticks, rotation = 45)
plt.xlim(-2, len(ticks)*sparsity-0.4)
plt.ylim(0.2, 1.0)
plt.ylabel('Dice Score')
plt.title('Mean dice score from different methods on selected regions')
plt.grid()
plt.tight_layout()
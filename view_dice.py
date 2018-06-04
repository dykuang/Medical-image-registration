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
after_sdn = np.load('dice_after_ver6.npy')
after_sdn_0 = np.load('dice_after_ver6_5.npy')
after_sdn_1 = np.load('dice_after_ver6_a1e-4.npy')
after_sdn_2 = np.load('dice_after_ver6_r1e-5.npy')
after_sdn_3 = np.load('dice_after_ver6_r2e-5.npy')
after_sdn_4 = np.load('dice_after_ver6_5r1e-5.npy')
after_sdn_5 = np.load('dice_after_ver6_r1e-6.npy')
after_sdn_6 = np.load('dice_after_ver6_5a2e-3.npy')
after_sdn_7 = np.load('dice_after_ver7.npy')
after_sdn_8 = np.load('dice_after_ver6_5r2e-5.npy')
after_sdn_9 = np.load('dice_after_5r5e-5.npy')
after_sdn_10 = np.load('dice_after_a5e-4r2e-5.npy')
after_sdn_11 = np.load('dice_after_2_0.npy')
after_sdn_12 = np.load('dice_after_0_1.npy')
after_sdn_13 = np.load('dice_after_1_2.npy')
after_sdn_14 = np.load('dice_after_branched.npy')
after_sdn_15 = np.load('dice_after_whole.npy')

after_utr = np.load('dice_after_utr.npy')

before_old_test = np.load('dice_before_testlist.npy')
after_old_test = np.load('dice_after_testlist.npy')
#plt.figure()
#plt.plot(before[0])
#plt.plot(after_utr[0])
#plt.plot(after_sdn[0])
#plt.legend(['Before', 'Utr', 'SDN'])
#plt.ylim([0.3, 0.9])

label_list = [i for i in range(21,35,1)] + [i for i in range(41,51,1)] + [i for i in range(61, 69, 1)] + [i for i in range(81,93,1)] + [101, 102, 121, 122] + [i for i in range(161, 167, 1)] + [181, 182]


def Dice_bar(Before, After_sdn, After_utr, labellist=label_list):
    # Setting the positions and width for the bars
    pos = list(range(len(labellist))) 
    width = 0.15 
        
    # Plotting the bars
    fig, ax = plt.subplots(figsize=(20,8))
    
    # Create a bar with pre_score data,
    # in position pos,
    plt.bar(pos, 
            #using df['pre_score'] data,
            before, 
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
    plt.bar([p + width for p in pos], 
            #using df['mid_score'] data,
            after_sdn,
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
    plt.bar([p + width*2 for p in pos], 
            #using df['post_score'] data,
            after_utr, 
            # of width
            width, 
            # with alpha 0.5
            alpha=0.5, 
            # with color
            color= 'r', 
            # with label the third value in first_name
            label=['after_utr']) 
    
    # Set the y axis label
    ax.set_ylabel('Dice')
    ax.set_xlabel('ROI label')
    # Set the chart's title
    ax.set_title('ROI Dices')
    
    # Set the position of the x ticks
    ax.set_xticks([p + 1.0 * width for p in pos])
    
    # Set the labels for the x ticks
    ax.set_xticklabels(label_list)
    
    # Setting the x-axis and y-axis limits
    plt.xlim(min(pos)-width, max(pos)+width*4)
    plt.ylim([0, 1.0])
    plt. xticks(rotation = 45)
    # Adding the legend and showing the plot
    plt.legend(['Before', 'After_sdn', 'After_utr'], loc='upper left')
    plt.grid()
    plt.show()
    

def Dice_box(data): # data is a list of 1d array
    fig = plt.figure(figsize=(9,6))
    ax = fig.add_subplot(111)
    ## to get fill color
    bp = ax.boxplot(data, patch_artist=True)
    
    ## change outline color, fill color and linewidth of the boxes
    for box in bp['boxes']:
        # change outline color
        box.set( color='#7570b3', linewidth=2)
        # change fill color
        box.set( facecolor = '#1b9e77' )
    
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
                  
    ax.set_xticklabels(['Before', 'After_sdn', 'After_utr'])
    ## Remove top axes and right axes ticks
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    
    
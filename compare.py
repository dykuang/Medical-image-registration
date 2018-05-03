# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 16:00:48 2018

@author: dykua

For comparison with AntsR
"""
import numpy as np
import os
import fnmatch
from skimage import io
from sklearn.metrics import jaccard_similarity_score   

def j_score(yTrue, yPred):
     js=[]
     for yT, yP in zip(yTrue, yPred):
          js.append(jaccard_similarity_score((yT>.1).flatten(), (yP>.1).flatten()))
     js = np.stack(js)
     return np.mean(js)

ants_output = r'datasets/FromAnts/'
target_path = r'datasets/oasis/'

ave_score = np.zeros(20)
for i in np.arange(20):
    brain_warped_files = fnmatch.filter(os.listdir(ants_output), '*to{}.tif'.format(i+1))
    brain_true = io.imread(os.path.join(target_path, os.listdir(target_path)[i]), as_grey = True)/255
    
    brain_warped_img = []
    for img in brain_warped_files:
        brain_warped_img.append(
                io.imread(os.path.join(ants_output, img), as_grey = True)/255)
        
    brain_warped_img = np.stack(brain_warped_img)
    
    brain_true_img =[brain_true for _ in range(len(brain_warped_img))]


    ave_score[i]=j_score(brain_true_img, brain_warped_img)
        
        
print(np.mean(ave_score))


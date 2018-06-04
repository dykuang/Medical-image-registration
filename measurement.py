# -*- coding: utf-8 -*-
"""
Created on Wed May  9 09:54:30 2018

@author: Dongyang
"""

import numpy as np
from sklearn.metrics import jaccard_similarity_score   
def j_score(yTrue, yPred):
     js=[]
     for yT, yP in zip(yTrue, yPred):
          js.append(jaccard_similarity_score((yT>0.1).flatten(), (yP>0.1).flatten()))
     js = np.stack(js)
     return np.mean(js)


def Dice(y_true, y_pred):
     T = (y_true.flatten()>0.1)
     P = (y_pred.flatten()>0.1)
     
     return 2*np.sum(T*P)/(np.sum(T) + np.sum(P))

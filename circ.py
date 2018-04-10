# -*- coding: utf-8 -*-
"""
Created on Sat Apr  7 17:44:12 2018

@author: dykua

Draw some basic geometry shape
"""

import numpy as np
from skimage.draw import circle, polygon
import matplotlib.pyplot as plt

res = 128
img = np.zeros((res, res), dtype=np.uint8)
rr, cc = circle(res/2, res/2, res/4)
img[rr, cc] = 1
plt.figure()
plt.imshow(img)


img = np.zeros((res, res), dtype=np.uint8)
vertx = [res*0.25, res*0.25, res*0.75, res*0.75]
verty = [res*0.25, res*0.75, res*0.75, res*0.25]
rr, cc = polygon(vertx, verty)
img[rr, cc] = 1
plt.figure()
plt.imshow(img)

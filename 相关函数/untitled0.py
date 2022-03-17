# -*- coding: utf-8 -*-
"""
Created on Sun May  5 10:57:42 2019

@author: 73707
"""

import numpy as np
mid=[1,2,3]
mid = np.array(mid)
filter3=[[[1,1,1],[1,1,1],[1,1,2]],[[2,2,2],[2,2,2],[2,2,2]],[[3,3,3],[3,3,3],[3,3,3]]]
filter3=np.array(filter3)
loss = np.zeros((3, 3))
for i in range(0,3):
    for j in range(0,3):
        loss[i][j]=( mid * filter3[:,i,j] ).sum()
#print (filter3.shape)
print (loss)

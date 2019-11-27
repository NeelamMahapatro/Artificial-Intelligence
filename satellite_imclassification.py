# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 09:22:13 2019

@author: student
"""

import cv2
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt

rimage = cv2.imread('rband.jpg')
gimage = cv2.imread('gband.jpg')
bimage = cv2.imread('bband.jpg')
iimage = cv2.imread('iband.jpg')

rimage = cv2.cvtColor(rimage, cv2.COLOR_BGR2GRAY)
gimage = cv2.cvtColor(gimage, cv2.COLOR_BGR2GRAY)
bimage = cv2.cvtColor(bimage, cv2.COLOR_BGR2GRAY)
iimage = cv2.cvtColor(iimage, cv2.COLOR_BGR2GRAY)

width = rimage.shape[1]
height = rimage.shape[0]


sum1=0
sum2=0
sum3=0
sum4=0

mean_shifted_matrix = np.zeros((4,width*height), dtype = float)

for k in range(4):
    for i in range(height):
        for j in range(width):
            if(k==0):
                sum1 = sum1+rimage[i][j]
            if(k==1):
                sum2 = sum2+gimage[i][j]
            if(k==2):
                sum3 = sum3+bimage[i][j]
            if(k==3):
                sum4 = sum4+iimage[i][j]
                
mean1 =sum1/(width*height)
mean2 =sum2/(width*height)
mean3 =sum3/(width*height)
mean4 =sum4/(width*height)
print(mean1, mean2, mean3, mean4)

count=0
for k in range(4):
    for i in range(height):
        for j in range(width):
            if(k==0):
                mean_shifted_matrix[0][count] = rimage[i][j]-mean1
            if(k==1):
                mean_shifted_matrix[1][count] = gimage[i][j]-mean2
            if(k==2):
                mean_shifted_matrix[2][count] = bimage[i][j]-mean3
            if(k==3):
                mean_shifted_matrix[3][count] = iimage[i][j]-mean4
'''
print("Mean Shifted  Matrix: \n")
print(mean_shifted_matrix)
'''
cov_matrix = np.cov(mean_shifted_matrix)

print("Covariance  Matrix: \n")
print(cov_matrix)
w, v = LA.eig(cov_matrix)

print("Eigen Vectors: \n")
print(v)

print("Eigen Values: \n")
print(w)

'''
maximum=w[3]
ans = 3
for i in range(3):
    if(w[i] > maximum):
        maximum=w[i]
        ans = i


sum_d = cov_matrix[0][0]+ cov_matrix[1][1] + cov_matrix[2][2] + cov_matrix[3][3]
x=0
for i in range(4):
    x = x+w[0]+w[1]+w[2]+w[3]

print(sum_d)
print(x)

if(ans==0):
    plt.imshow(rimage, cmap='gray')
if(ans==1):
    plt.imshow(gimage, cmap='gray')
if(ans==2):
    plt.imshow(bimage, cmap='gray')
if(ans==3):
    plt.imshow(iimage, cmap='gray')
'''
rowFeatureVector = np.transpose(v)
finalData = np.dot(rowFeatureVector, mean_shifted_matrix)

fimage1 = np.zeros((width,height), dtype = int)
fimage2 =  np.zeros((width,height), dtype = int)
fimage3 = np.zeros((width,height), dtype = int)
fimage4 =  np.zeros((width,height), dtype = int)

count=0

for k in range(4):
    for i in range(height):
        for j in range(width):
            if(k==0):
                mean_shifted_matrix[0][count] = rimage[i][j]-mean1
            if(k==1):
                mean_shifted_matrix[1][count] = gimage[i][j]-mean2
            if(k==2):
                mean_shifted_matrix[2][count] = bimage[i][j]-mean3
            if(k==3):
                mean_shifted_matrix[3][count] = iimage[i][j]-mean4
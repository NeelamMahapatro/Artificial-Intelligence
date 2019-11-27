# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 10:50:05 2019

@author: user
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 09:22:13 2019

@author: student
"""

import cv2
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
from os import listdir
from PIL import Image as Image
import operator
import math

def loadImages(path):
    imagesList = listdir(path)
    loadedImages = []
    for image in imagesList:
        img = cv2.imread(path + image)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = np.asarray(img)
            loadedImages.append(img)

    return loadedImages

path = "./dataset/Train/"

imgs = loadImages(path)
'''
for img in imgs:
    plt.imshow(img, cmap='gray')
    plt.show()
'''

image_matrix = np.zeros((len(imgs), 425*425), dtype=np.uint8)
print(len(imgs))
k=0
for img in imgs:
    image_matrix[k] = imgs[k].flatten() 
    k = k+1

mean_vec = np.zeros((len(imgs),1), dtype=float)
for i in range(len(imgs)):
    for j in range(imgs[i].shape[0] * imgs[i].shape[1]):
        mean_vec[i] = mean_vec[i]+image_matrix[i][j]
    mean_vec[i] = mean_vec[i] / (imgs[i].shape[0] * imgs[i].shape[1])
    
mean_shift = image_matrix - mean_vec
cov_matrix = np.cov(mean_shift)


eigen_val, eigen_vec = LA.eig(cov_matrix)

map_eigen = dict()
for i in range(len(eigen_val)):
   map_eigen[eigen_val[i]] = eigen_vec[i]

sorted_dic = dict(sorted(map_eigen.items(), key = operator.itemgetter(0), reverse = True))
print(np.shape(eigen_vec))

p = 25
sorted_eigen = np.zeros((p, k), dtype = float)

k1=0
for i in sorted_dic:
    sorted_eigen[k1] = sorted_dic[i]
    k1=k1+1
    if(k1>=p):
        break

print(sorted_eigen)

sorted_eigen_t = np.transpose(sorted_eigen)
 
eigen_faces = np.zeros((k,425*425), dtype=float)

eigen_faces = np.dot(sorted_eigen, mean_shift)
print("Eigen Face")
print(np.shape(eigen_faces))

sign_faces = np.dot(eigen_faces, np.transpose(mean_shift))
print("Sign Face")
print(np.shape(sign_faces))


path1 = "./dataset/Test/"

test_imgs = loadImages(path1)
test_image_matrix = np.zeros((len(test_imgs), 425*425), dtype=np.uint8)

k=0
for img in test_imgs:
    test_image_matrix[k] = test_imgs[k].flatten() 
    k = k+1


mean_vec_test = np.zeros((len(test_imgs),1), dtype=float)
for i in range(len(test_imgs)):
    for j in range(test_imgs[i].shape[0] * test_imgs[i].shape[1]):
        mean_vec[i] = mean_vec[i]+test_image_matrix[i][j]
    mean_vec[i] = mean_vec[i] / (imgs[i].shape[0] * imgs[i].shape[1])
    
mean_shift_test = test_image_matrix - mean_vec_test
print("mean shift test")
print(np.shape(mean_shift_test))

project_testface = np.dot(eigen_faces,np.transpose(mean_shift_test))
print("Project Test Face")
print(np.shape(project_testface))

error = np.zeros((1, 25), dtype=float) 
for i in range(25):
    for j in range(p):
        error[0][i] = error[0][i]+(sign_faces[j][i]-project_testface[j][0])**2
for i in range(25):
    error[0][i] = math.sqrt(error[0][i]/180625)

print(error)


        
        
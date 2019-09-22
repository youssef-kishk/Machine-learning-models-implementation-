# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 00:19:11 2019

@author: Youssef
"""

import numpy as np
from scipy.io import loadmat 
import matplotlib.pyplot as plt
import scipy.linalg 

def pca(img):
    img = (img - img.mean()) / img.std()
    img = np.matrix(img)
    cov = (img.T * img) / img.shape[0]
    eigen_vectors,eigen_values,vt = scipy.linalg.svd(cov,full_matrices = True)
    
    return eigen_vectors,eigen_values
#*******************************************************************************************#

def project_data(eigen_values, eigen_vectors, k):
    U_reduced = eigen_vectors[:,:k]
    return np.dot(eigen_values, U_reduced)
#*******************************************************************************************#
    
def recover_data(Z, eigen_vectors, k):
    U_reduced = eigen_vectors[:,:k]
    return np.dot(Z, U_reduced.T)
#*******************************************************************************************#    
if __name__ == '__main__':
    #load data
    imgs = loadmat('ex7faces.mat')['X']
    
    test_img = imgs[1000,:]
    plt.imshow(np.reshape(test_img,(32,32)))
    plt.show()
    
    #determine eigen values and vectors matrices
    eigen_vectors,eigen_values = pca(test_img)
    
    
    # new_img = (eigen vectors).T * (eigen values) * (eigen vectors)
    Z = project_data(eigen_values, eigen_vectors, 1)
    X_recovered = recover_data(Z, eigen_vectors, 1)
    
    plt.imshow(np.reshape(X_recovered,(32,32)))
    plt.show()
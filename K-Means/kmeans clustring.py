# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 17:17:50 2019

Unsupervised learning K-means clustring
@author: Youssef Kishk
"""
import numpy as np
from scipy.io import loadmat 
import matplotlib.pyplot as plt


#compute initial random centroids for the clusters
def compute_initial_centroids(data,K,m,n):
    centroids = np.zeros((K,n))
    for i in range(K):
         random = np.random.randint(0,m)
         centroids[i,:] = data[random,:]
    return centroids

#*******************************************************************************************#
    
#map each point to its clusters according to its clossest centroid 
def find_clusters(data,K,centroids,m):
    clusters = np.zeros(m)
    
    for i in range(m):
        min_value = 100000000
        for j in range(K):
            distance = np.sum((data[i,:]-centroids[j,:])**2)
            if distance < min_value:
                clusters[i] = j
                min_value = distance
    return clusters

#*******************************************************************************************# 
    
#compute the values of new centroids of rach cluster
def find_new_centroids(data,K,clusters,m,n):
    new_centroids = np.zeros((K,n))
    
    for i in range(K):
        temp_indeces = np.where(clusters == i)
        new_centroids[i,:] = (np.sum(data[temp_indeces,:],axis = 1) / len(temp_indeces[0]))
    return new_centroids

#*******************************************************************************************#
    
# main K-means function
def Kmeans(data,centroids,iterations,n,m,K):
    clusters = np.zeros(m)
    for i in range(iterations):
        clusters = find_clusters(data,K,centroids,m)
        centroids = find_new_centroids(data,K,clusters,m,n)
    return clusters,centroids

#*******************************************************************************************#
    
def read_data(path):
    data = loadmat(path)['X']
    return data
#*******************************************************************************************#
    
def plot_input_data(initial_data):
    fig, ax = plt.subplots(figsize=(10,6))
    ax.scatter(initial_data[:,0], initial_data[:,1], s=8, c='b', marker='o')
    ax.set(xlabel='X0', ylabel='X1' )
    ax.set(title='Input data')
#*******************************************************************************************#
   
if __name__ == '__main__':
    colors = ['r','g','b','Orange','Black','Yellow','Aqua']
    #load data
    data = read_data('ex7data2.mat')
    #num of rows
    m = data.shape[0]
    #num of features
    n = data.shape[1]
    
    #graph the loaded input data
    plot_input_data(data)
    
    # Read number of required clusters as an input
    print('Number of clusters:',end = '')
    K = int(input())
    
    #initialize centroids and clusters
    initial_centroids = compute_initial_centroids(data,K,m,n)
    initial_clusters = find_clusters(data,K,initial_centroids,m)
    initial_centroids = find_new_centroids(data,K,initial_clusters,m,n)
    
    
    # Read number of required steps as an input
    print('Number of iterations:',end = '')
    num_of_iterations = int(input())
    
    #apply kmeans for certain number of iterations
    clusters,centroids = Kmeans(data,initial_centroids,num_of_iterations,n,m,K)
        
    
        
    #graph the loaded input data
    fig, ax = plt.subplots(figsize=(10,6)) 
    ax.set(title='clustered data')
    ax.set(xlabel='X0', ylabel='X1') 
    for i in range (K):
         x = data[np.where(clusters == i)[0],:]
         #scatter the data
         ax.scatter(x[:,0], x[:,1], s=15, c=colors[i], marker='o')
         #plot the final centroids
         ax.scatter(centroids[i,0], centroids[i,1], s=700, c=colors[i], marker='+')
#*******************************************************************************************#


    
    
    
    
    
    
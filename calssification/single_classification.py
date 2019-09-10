# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 15:14:04 2019

@author: Youssef Kishk
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))
#*******************************************************************************************#
def cost(theta,x,y):
    num_of_rows = len(x)
    theta = np.matrix(theta)
    term1 = np.multiply(y,np.log(sigmoid(x * np.transpose(theta))))
    term2= np.multiply(1-y,np.log(1- sigmoid(x * (theta.T))))
    temp = np.sum(term1+term2)
    cost_val = (-1*temp) / num_of_rows
    
    return cost_val
#*******************************************************************************************#
def gradient_descent(theta,x,y):
    num_of_rows = len(x)
    theta = np.matrix(theta)
    num_of_thetas = theta.shape[1]
   #new_thetas=np.zeros(num_of_thetas)
    new_thetas=np.matrix(np.zeros(theta.shape))
    
    err = sigmoid(x*theta.T) - y
    for i in range(num_of_thetas):
        temp = np.multiply(err,x[:,i])
        new_thetas[0,i]= np.sum(temp) / num_of_rows
    return new_thetas
#*******************************************************************************************#
def predict(theta,x):
    theta = np.matrix(theta)
    prob = sigmoid(x*theta.T)
   
    return [1 if i>=0.5 else 0 for i in prob]
#*******************************************************************************************#
def main():
    #read data
    data = pd.read_csv("data.txt",names=["x1","x2","y"])
    
    
    #add ones column before data for theta 0 calculations
    data.insert(0,'Ones',1)
    
    
    num_of_cols=data.shape[1]
    
    #separate postive from negative values
    postive= data[data['y'].isin([1])]
    negative= data[data['y'].isin([0])]
    
    #plot data
    fig, ax = plt.subplots(figsize=(5,5))
    ax.scatter(postive['x1'], postive['x2'], s=50, c='b', marker='o',label='one')
    ax.scatter(negative['x1'], negative['x2'], s=50, c='r', marker='*',label='zero')
    
    
    #split data form label (x,y)
    x=data.iloc[:,0:num_of_cols-1]
    y=data.iloc[:,num_of_cols-1:num_of_cols]
    
    
    #convert to numpy metrices
    x=np.matrix(x.values)
    y=np.matrix(y.values)
    theta=np.matrix(np.zeros(x.shape[1]))
    
    
    import scipy.optimize as opt
    #best theta's values for minimum cost
    result_thetas = opt.fmin_tnc(func=cost,x0=theta,fprime=gradient_descent,args=(x,y))[0]

    #predict the values 
    predictions = predict(result_thetas,x)
    
    
    correct=0
    for i in range(len(y)):
        if y[i]==predictions[i]:
            correct+=1
    print('accuracy = ',(correct/len(y))*100)
 
    
if __name__ == '__main__':
    main()
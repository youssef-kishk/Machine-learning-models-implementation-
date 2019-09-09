# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 16:37:08 2019

@author: Youssef
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#cost function (goal to be minimized)
def cost(x,y,theta,num_of_rows):
    cost_function = np.power((x*theta.T)-y,2) 
    return (np.sum(cost_function)/(2*num_of_rows))

#******************************************************************************************************#
def gradient_descent(x,y,theta,alpha,iterations,num_of_rows):
    new_theta=np.matrix(np.zeros(theta.shape))
    num_of_thetas = theta.shape[1]
    cost_values=np.zeros(iterations)
    
    
    for i in range(iterations):
        err = ((x*theta.T))-y
        for j in range(num_of_thetas):
            temp = np.multiply(err,x[:,j])           
            new_theta[0,j]=theta[0,j] - ((alpha/num_of_rows) * np.sum(temp))
    
        theta=new_theta
        
        cost_values[i] = cost(x,y,theta,num_of_rows)
        
    return theta,cost_values
        
#******************************************************************************************************#            
    
def main(alpha,iterations):
   #read data
   data = pd.read_csv("data.txt",names=["population","profit"])
   print(data.describe())
     
   #add ones column before data for theta 0 calculations
   data.insert(0,'Ones',1)
          
   num_of_cols=data.shape[1]
   num_of_rows=data.shape[0]
   
   #split data form label (x,y)
   x=data.iloc[:,0:num_of_cols-1]
   y=data.iloc[:,num_of_cols-1:num_of_cols]
    
   #convert to numpy metrices
   x=np.matrix(x.values)
   y=np.matrix(y.values)
   
   
   #theta0,1
   theta=np.matrix(np.array([0,0]))
   
   #apply gradient descent
   theta,cost_values=gradient_descent(x,y,theta,alpha,iterations,num_of_rows)
  
   #best fit line function
   line_function = theta[0,0]+ (theta[0,1]*x)
   
   fig, ax = plt.subplots(figsize=(5,5))
   ax.plot(x, line_function)
   ax.scatter(data.population,data.profit)
   ax.set_xlabel('Population')
   ax.set_ylabel('Profit')
   ax.set_title("Best fit line")
   
   #find cost function graph
   fig, ax = plt.subplots(figsize=(5,5))
   ax.plot(np.arange(iterations), cost_values)
   ax.set_xlabel('Iterations')
   ax.set_ylabel('Cost')
   
#******************************************************************************************************#   
if __name__ == '__main__':
    #specify alpha and number of iterations
    alpha=0.01
    iterations=10000
    main(alpha,iterations)





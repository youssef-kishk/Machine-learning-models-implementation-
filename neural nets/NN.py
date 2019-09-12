# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 16:02:44 2019

@author: Youssef Kishk
"""

import numpy as np


def sigmoid(z):
    return 1/(1+np.exp(-z))

def derivative_simoid(z):
    return z * (1.0 - z)

#***********************************************************************************************#
class neural_network:
    def __init__(self,x,y):
        self.input = x
        
        self.input_size = self.input.shape[1]       
        self.output_size = 1
        self.hidden_layer_size = self.input_size+1
        
        
        self.weights1 = np.random.randn(self.input_size,self.hidden_layer_size)      
        self.weights2 = np.random.randn(self.hidden_layer_size,  self.output_size)
        
        self.y=y
        self.output = np.zeros(self.y.shape)
         
    def forward_prob(self):
        #a's values of first hidden layer
        self.layer1 = sigmoid(np.dot(self.input,self.weights1))
        
        self.output = sigmoid(np.dot(self.layer1,self.weights2))
          
    def back_prob(self):
        error2 = self.y - self.output
        delta2 = error2 * derivative_simoid(self.output)
        updated_weights2 = np.dot(self.layer1.T,delta2)
        
        
        error1 = np.dot(delta2,self.weights2.T)
        delta1 = error1 * derivative_simoid(self.layer1)
        updated_weights1 = np.dot(self.input.T,delta1)
        
        self.weights1 +=updated_weights1
        self.weights2 +=updated_weights2
        
    def saveWeights(self):
        np.savetxt("w1.txt", self.weights1, fmt="%s")
        np.savetxt("w2.txt", self.weights2, fmt="%s")
#***********************************************************************************************#
if __name__ == '__main__':    
    x = np.array([[0,0,1],
                  [0,1,1],
                  [1,0,1],
                  [1,1,1]])
    
    y = np.array([[0],
                  [1],
                  [1],
                  [0]])
    
    nn = neural_network(x,y)
    epochs = 1000
    
    print ("Initial Loss: \n" + str(np.mean(np.square(y - nn.output))))
    for i in range(epochs):
        nn.forward_prob()
        nn.back_prob()
        
    print('True op = \n',y)
    print('predicted op = \n',nn.output)    
    print ("Final Loss: \n" + str(np.mean(np.square(y - nn.output))))
    
    nn.saveWeights()
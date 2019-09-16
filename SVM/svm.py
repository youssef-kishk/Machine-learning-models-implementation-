# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 21:23:30 2019

    Spam or non spam mails 
@author: Youssef Kishk
"""
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from scipy.io import loadmat
from sklearn import svm

if __name__ == '__main__':
    spam_train = loadmat('spamTrain.mat')
    spam_test = loadmat('spamTest.mat')

    
    
    x_train = spam_train['X']
    X_test = spam_test['Xtest']
    #ravel convert from A column-vector to a 1d array
    y_train = spam_train['y'].ravel()
    y_test = spam_test['ytest'].ravel()
    
    
    #small C value cause under fitting
    #large C value cause over fitting
    #kernel types : linear or rbf 
    svm = svm.SVC(C=1.0, kernel='linear')
    #Train data
    svm.fit(x_train,y_train)

    #F1 score
    print('Test accuracy = ',svm.score(X_test, y_test)*100)
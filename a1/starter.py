import tensorflow as tf
import numpy as np 
import matplotlib.pyplot as plt

def loadData():
    with np.load('notMNIST.npz') as data :
        Data, Target = data ['images'], data['labels']
        posClass = 2
        negClass = 9
        dataIndx = (Target==posClass) + (Target==negClass)
        Data = Data[dataIndx]/255.
        Target = Target[dataIndx].reshape(-1, 1)
        Target[Target==posClass] = 1
        Target[Target==negClass] = 0
        np.random.seed(421)
        randIndx = np.arange(len(Data))
        np.random.shuffle(randIndx)
        Data, Target = Data[randIndx], Target[randIndx]
        trainData, trainTarget = Data[:3500], Target[:3500]
        validData, validTarget = Data[3500:3600], Target[3500:3600]
        testData, testTarget = Data[3600:], Target[3600:]
    return trainData, validData, testData, trainTarget, validTarget, testTarget

#Input Notes: W = dx1, b=1x1, x=Nxd, y=Nx1, reg=1x1 ,where N = # of data points, d = # of features

def MSE(W, b, x, y, reg):
    #MSE loss function
    N = len(y)
    for i in range(N):
        W = W.flatten()
        x_i = x[i:1]
        x_i = x_i.flatten()
        loss += np.square(np.dot(W,x_i)+b-y[i]) #x[i:1] is row vector, x = 1xd
    loss *= 2/N
    
    #Regularization
    regular = reg/2*np.dot(W,np.transpose(W))
    
    #Return Total Loss
    return loss+regular

def gradMSE(W, b, x, y, reg):
    # Your implementation here
    N=len(y)
    for i in range(N):
        W = W.flatten()
        x_i = x[i:1]
        x_i = x_i.flatten()
        grad_W += (np.dot(W,x_i)-b-y[i])*x_i #grad_w = 1xd array
        grad_b += np.dot(W,x_i)-b-y[i] #grad_b = 1x1
    grad_W *= 1/N
    grad_W += reg*np.flatten(W)
    grad_b *= 1/N
    
    return np.transpose([grad_W]), grad_b

def crossEntropyLoss(W, b, x, y, reg):
    # Your implementation here
    N = len(y)
    for i in range(N):
        x_i = x[i:1]
        x_i=x_i.flatten()
        y_output = sigmoid(W,x[i:1],b)
        loss += -1*y[i]*np.log(y_output)-(1-y[i])*np.log(1-np.log(y_output)) 
    loss *= 1/N
    
    #Regularization
    regular = reg/2*np.dot(W,np.transpose(W))
    
    #Return Total Loss
    return loss+regular

def sigmoid(W,x_i,b):
    z=np.dot(x_i,W)+b
    y_output = 1/(1+np.exp(-1*z))
    
    return y_output #return a constant

def gradCE(W, b, x, y, reg):
    # Your implementation here
    

def grad_descent(W, b, trainingData, trainingLabels, alpha, iterations, reg, EPS):
    # Your implementation here

def buildGraph(beta1=None, beta2=None, epsilon=None, lossType=None, learning_rate=None):
    # Your implementation here


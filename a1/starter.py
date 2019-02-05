#%% Import Modules
import tensorflow as tf
import numpy as np 
from numpy import linalg as LA
#import matplotlib.pyplot as plt

#%% Load Data
def loadData():
    with np.load('/Users/matt/Documents/Winter_2019/ECE421/ML_Assignments/a1/notMNIST.npz') as data:
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
        #print(trainData.shape) #(3500,28,28)
        #print(validData.shape) #(100,28,28)
        #print(testData.shape) #(145,28,28)
        #print(trainTarget.shape) #(3500,1)
        #print(validTarget.shape) #(100,1)
        #print(testTarget.shape) #145,1)
    return trainData, validData, testData, trainTarget, validTarget, testTarget

trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()
#Input Notes:
#N = # of data points = 3500, d = # of features = 28x28 = 784
#Train Data: W = dX1, b=1x1, x=Nxd, y=Nx1, reg=1x1

#%% MSE Gradient and Loss Functions
def MSE(W, b, x, y, reg):
    #Declare Variables
    N = len(y)
    d = x.shape[1]*x.shape[2]
    loss=0
    
    #Convert x into 2D matrix (Nxd)
    x = np.reshape(x,(N,d))
    #W = np.reshape(W,(d,1))
    #loss = LA.norm(np.dot(x,W)+b-y)
    W = W.flatten()
    for i in range(N):
        loss += np.square(np.dot(W,x[i])+b-y[i]) #x[i:1] is row vector, x = 1xd
    loss *= 1/(2*N)
    
    #Regularization
    #regular = reg/2*np.square(LA.norm(W))
    regular = reg/2*np.dot(W,W)
    
    #Return Total Loss
    return loss+regular

def gradMSE(W, b, x, y, reg):
    # Your implementation here
    N=len(y)
    grad_W=0
    grad_b=0
    x = np.reshape(x,(N,x.shape[1]*x.shape[2]))
    W = W.flatten()
    for i in range(N):
        grad_W += (np.dot(W,x[i])+b-y[i])*x[i] #grad_w = 1xd array
        grad_b += np.dot(W,x[i])+b-y[i] #grad_b = 1x1
    grad_W *= 1/N
    grad_W += reg*W
    
    grad_b *= 1/N
    
    return grad_W, grad_b
#%% MSE Gradient and Loss Functions Testing
    W = np.zeros(testData.shape[1]*testData.shape[2])
    reg = 0
    b=1
    print(MSE(W,b,testData,testTarget,reg))
    grad_W_test, grad_b_test = gradMSE(W, b, testData,testTarget, reg)
    #print(gradMSE(W,b,testData,testTarget,reg))

#%% Cross Entropy Gradient and Loss Functions
def sigmoid(W,x_i,b):
    return 1/(1+np.exp(-np.dot(x_i,W)-b))

def crossEntropyLoss(W, b, x, y, reg):
    # Your implementation here
    N = len(y)
    loss=0
    x = np.reshape(x,(N,x.shape[1]*x.shape[2]))
    W = W.flatten()
    for i in range(N):
        y_output = sigmoid(W,x[i],b)
        loss -= y[i]*np.log(y_output)+(1-y[i])*np.log(1-y_output) 
    loss *= 1/N
    
    #Regularization
    regular = reg/2*np.dot(W,W)
    
    #Return Total Loss
    return loss+regular

def gradCE(W, b, x, y, reg):
    # Your implementation here
    N=len(y)
    grad_W=0
    grad_b=0
    x = np.reshape(x,(N,x.shape[1]*x.shape[2]))
    W = W.flatten()
    for i in range(N):
        y_output=sigmoid(W,x[i],b)
        grad_b += y_output-y[i]
        grad_W += grad_b*x[i]
    grad_W *= 1/N
    grad_W += reg*W
    
    grad_b *= 1/N
    
    return grad_W, grad_b
    
#%% CrossEntropy Gradient and Loss Functions Testing
    W = np.zeros(testData.shape[1]*testData.shape[2])
    reg = 0.1
    b=1
    print(crossEntropyLoss(W,b,testData,testTarget,reg))
    print(gradCE(W,1,testData,testTarget,reg))
    
    
#%% Gradient Descent
def grad_descent(W, b, trainingData, trainingLabels, alpha, iteration, reg, EPS, lossType=None):
    # Your implementation here
    prev_W = np.ones_like(W)
    if (lossType == "MSE"):
        for epoch in range(iteration):
            loss = MSE(W,b,testData,testTarget,reg)
            delta_W = LA.norm(W-prev_W)
            print("Delta W = "+str(delta_W)+" Loss = "+str(loss)+'\n')
            if (delta_W <= EPS):
                return W,b
            else:
                prev_W = np.copy(W)
                grad_W, grad_b = gradMSE(W,b,testData,testTarget,reg)
                W -= alpha*grad_W
                b -= alpha*grad_b
        return W, b
            
    elif (lossType == "CE"):
        for epoch in range(iteration):
            loss = crossEntropyLoss(W,b,testData,testTarget,reg)
            delta_W = LA.norm(W-prev_W)
            print("Delta W = "+str(delta_W)+" Loss = "+str(loss)+'\n')
            if (delta_W <= EPS):
                return W,b
            
            else:
                prev_W = np.copy(W)
                grad_W, grad_b = gradCE(W,b,testData,testTarget,reg)
                W -= alpha*grad_W
                b -= alpha*grad_b
        return W, b
     
#%% Gradient Descent Test
    W = np.zeros(trainData.shape[1]*trainData.shape[2])
    b = 0 
    alpha = 0.001
    epochs = 5000
    reg = 0 
    EPS = 10e-7 # 1x10^-7
    W_optimal,b_optimal = grad_descent(W, b, trainData, trainData, alpha, epochs, reg, EPS,"CE")

def ModelTest(W,b,testData,testTarget,lossType=None):
    correct=0
    N = len(testData)
    d = testData.shape[1]*testData.shape[2]
    x_test = np.reshape(testData,(N,d))
    y_predict=np.empty(N)
    if (lossType == "MSE"):
        for i in range(N):
            y_predict[i] = np.dot(W,x_test[i])+b
            if(abs(y_predict[i] - testTarget[i])<0.5):
                correct+=1
    elif (lossType == "CE"):
        for i in range(N):
            y_predict[i] = sigmoid(W,x_test[i],b)
            if(abs(y_predict[i] - testTarget[i])<0.05):
                correct+=1
        
    print("The Model is "+str(100*correct/N)+"% accurate.\n")
    
    return 100*correct/N, y_predict

accurary, y_predict = ModelTest(W_optimal,b_optimal,testData,testTarget, "CE")

#%% buildGraph and SGD
def buildGraph(lossType=None):
    # Your implementation here
    tf.set_random_seed(421)
    
    #Initialize Parameters
    learning_rate = 0.001
    iterations = 700
    batch = 500 
    d = np.shape(testData.shape[1]*testData.shape[2])
    regularization = 0
    
    #Initialize X,Y, weight and bias, reg
    X = tf.placeholder("float")
    Y = tf.placeholder("float")
    
    W = tf.random.truncated_normal(shape=(batch,d),stddev=0.5,seed=421,name = "Weight")
    b = tf.random.truncated_normal(shape=(1,1),stddev=0.5,seed=421,name = "Bias")
    reg = tf.constant()
    
    #Initialize Prediction Model and Loss function
    if (lossType == "MSE"):
        pred = tf.add(tf.multiply(X, W), b)
        loss = tf.losses.mean_squared_error(labels=Y,predictions=pred)
        
        
    elif(lossType == "CE"):
        pred = tf.sigmoid(tf.add(tf.multiply(X, W), b))
        loss = tf.losses.sigmoid_cross_entropy(labes=Y,predicitons=pred)
    
    #Initiaze optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate)
        
    #Initialize the variables
    init = tf.global_variables_initializer()
    
    #Start Training Session
    with tf.Session() as sess:
        #Run the initializer
        sess.run(init)
        
        for epoch in range(iterations):
            for (x, y) in (testData, targetData):
                sess.run(optimizer.minimize(loss), feed_dict={X: trainData, Y: trainTarget})
                if (epoch+1) % 50 == 0:
                    #print epoch and Loss
                    current_loss = sess.run(loss, feed_dict={X: trainData, Y: trainTarget})
                    print("Epoch: ",epoch, "Loss= ",current_loss)

#%%SGD Implementation

def SGD()
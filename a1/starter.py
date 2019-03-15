#%% Import Modules
import tensorflow as tf
import numpy as np 
from numpy import linalg as LA
import matplotlib.pyplot as plt

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
    W = W.flatten()
    for i in range(N):
        loss += np.square(np.dot(W,x[i])+b-y[i]) #x[i:1] is row vector, x = 1xd
    loss *= 1/(2*N)
    
    #Regularization
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
def grad_descent(W, b, Data, Target, alpha, iteration, reg, EPS, lossType=None):
    # Your implementation here
    prev_W = np.ones_like(W)
    if (lossType == "MSE"):
        for epoch in range(iteration):
            loss = MSE(W,b,Data,Target,reg)
            delta_W = LA.norm(W-prev_W)
            print("Delta W = "+str(delta_W)+" Loss = "+str(loss)+'\n')
            if (delta_W <= EPS):
                return W,b
            else:
                prev_W = np.copy(W)
                grad_W, grad_b = gradMSE(W,b,Data,Target,reg)
                W -= alpha*grad_W
                b -= alpha*grad_b
        return W, b
            
    elif (lossType == "CE"):
        for epoch in range(iteration):
            loss = crossEntropyLoss(W,b,Data,Target,reg)
            delta_W = LA.norm(W-prev_W)
            print("Delta W = "+str(delta_W)+" Loss = "+str(loss)+'\n')
            if (delta_W <= EPS):
                return W,b
            
            else:
                prev_W = np.copy(W)
                grad_W, grad_b = gradCE(W,b,Data,Target,reg)
                W -= alpha*grad_W
                b -= alpha*grad_b
        return W, b
     
#%% Gradient Descent Test
    W = np.zeros(trainData.shape[1]*trainData.shape[2])
    b = 1
    alpha = 0.001
    epochs = 5000
    reg = 0 
    EPS = 1e-7 # 1x10^-7
    lossType = "CE"
    W_optimal,b_optimal = grad_descent(W, b, trainData, trainData, alpha, epochs, reg, EPS,lossType)

    N = len(testData)
    d = testData.shape[1]*testData.shape[2]
    x_test = np.reshape(testData,(N,d))
    y_predict=np.empty(N)
    correct=0
    if (lossType == "MSE"):
        for i in range(N):
            y_predict[i] = np.dot(W,x_test[i])+b
            if(round(y_predict[i]) - testTarget[i]==0):
                correct+=1
    elif (lossType == "CE"):
        for i in range(N):
            y_predict[i] = sigmoid(W,x_test[i],b)
            if(round(y_predict[i]) - testTarget[i]==0):
                correct+=1
    
    accuracy = 100*correct/N
    print("The Model is "+str(accuracy)+"% accurate.\n")
#%% buildGraph and SGD
def buildGraph(beta1=None, beta2=None, epsilon=None, lossType=None, learning_rate=None):
    # Your implementation here    
    #Reset to defaultgraph
    tf.reset_default_graph()
    
    #Initialize Parameters
    learning_rate = 0.001
    d = testData.shape[1]*testData.shape[2]
    Lambda = 0
    
    #Initialize Placeholders and Constants
    X = tf.placeholder("float")
    Y = tf.placeholder("float")
    reg = tf.constant(Lambda, name = "Lambda")
    
    #Initialize Variables
    tf.set_random_seed(421)
    W = tf.get_variable(initializer = tf.truncated_normal((d,1),stddev=0.5,seed=421),name = "Weight")
    b = tf.get_variable(initializer = tf.truncated_normal((1,1),stddev=0.5,seed=421),name = "Bias")
    
    
    #Initialize Prediction Model and Loss function
    if (lossType == "MSE"):
        pred = tf.add(tf.matmul(X, W), b)
        loss = tf.losses.mean_squared_error(labels=Y,predictions=pred)
        correct_prediction = tf.equal(tf.round(pred), Y)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        
        
    elif(lossType == "CE"):
        logit =tf.add(tf.matmul(X, W), b)
        pred = tf.sigmoid(logit)
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=Y,logits=logit))
        correct_prediction = tf.equal(tf.round(pred), Y)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    #Initiaze optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate, name='ADAM').minimize(loss)
        
    return W, b, pred, X, Y, loss, accuracy, optimizer, reg 
#%%SGD Implementation
def SGD(batchSize,iterations,lossType=None):
    #Load and Reshape Data
    trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()
    d = trainData.shape[1]*trainData.shape[2]
    trainData = np.reshape(trainData,(trainData.shape[0],d))
    validData = np.reshape(validData,(validData.shape[0],d))
    testData = np.reshape(testData,(testData.shape[0],d))
    
    #Caluclate # of Batchs
    batches = trainData.shape[0]//batchSize
    
    #Initiate Variables
    if (lossType == "MSE"):
        W, b, pred, X, Y, loss, accuracy, optimizer, reg  = buildGraph(beta1=0.9, beta2=0.999, epsilon=1e-08, lossType="MSE", learning_rate=0.001)
    elif (lossType == "CE"):
        W, b, pred, X, Y, loss, accuracy, optimizer, reg  = buildGraph(beta1=0.9, beta2=0.999, epsilon=1e-08, lossType="CE", learning_rate=0.001)
        
    trainLoss,trainAcc,validLoss,validAcc,testLoss,testAcc = ([] for i in range(6))
    
    #Start Training Session
    with tf.Session() as sess:
        #Run the initializer
        init = tf.global_variables_initializer()
        sess.run(init)
        
        for epoch in range(iterations):
            #Create mini-batches
            rand_int = np.random.choice(trainData.shape[0],size=batchSize)
            x_batch = np.empty((batchSize,d))
            y_batch = np.empty((batchSize,1))
            for i in range(batchSize):
                index = rand_int[i]
                x_batch[i] = trainData[index]
                y_batch[i] = trainTarget[index]
            
            #Run sessions and store losses
            _,temp_trainLoss,temp_trainAcc = sess.run([optimizer,loss,accuracy], feed_dict={X: x_batch, Y: y_batch})
            trainLoss.append(temp_trainLoss)
            trainAcc.append(temp_trainAcc)
            print("Epoch: {0}, train loss: {1:.2f}, train accuracy: {2:.01%}". format(epoch + 1, temp_trainLoss, temp_trainAcc))
            
            temp_validLoss,temp_validAcc = sess.run([loss,accuracy], feed_dict={X: validData, Y: validTarget})
            validLoss.append(temp_validLoss)
            validAcc.append(temp_validAcc)
            
            temp_testLoss,temp_testAcc = sess.run([loss,accuracy], feed_dict={X: testData, Y: testTarget})
            testLoss.append(temp_testLoss)
            testAcc.append(temp_testAcc)
            
        #Save Session`
        W_optimal, b_optimal = sess.run([W,b], feed_dict={X: testData, Y: testTarget})
        
        print("Optimization finished!")
        return W_optimal, b_optimal, trainLoss, validLoss, testLoss, trainAcc, validAcc, testAcc
        
#%%SGD Testing
    #Run SGD algorithm
    epochs = 5000
    batchSize = 500
    lossType = "MSE"
    W_optimal,b_optimal, SGD_trainLoss,SGD_validLoss, SGD_testLoss, SGD_trainAcc, SGD_validAcc, SGD_testAcc= SGD(batchSize,epochs,lossType)
   
    #Plot SGD results
    x_axis = np.arange(epochs)+1
    
    plt.figure(figsize=(10,10))
    
    plt.subplot(211)
    plt.plot(x_axis,SGD_trainLoss,color='c',linewidth=2.0,label="Training")
    plt.plot(x_axis,SGD_validLoss,color='b',linewidth=2.0,label="Validation")
    plt.plot(x_axis, SGD_testLoss,color='g',linewidth=2.0,label="Test")
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.title("Stoichastic Gradient Descent to Minimize " + lossType)
    plt.legend()

    
    plt.subplot(212)
    plt.plot(x_axis,SGD_trainAcc,color='c',linewidth=2.0,label="Training")
    plt.plot(x_axis,SGD_validAcc,color='b',linewidth=2.0,label="Validation")
    plt.plot(x_axis, SGD_testAcc,color='g',linewidth=2.0,label="Test")
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.legend()
    
    plt.show()
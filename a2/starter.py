#%%Import Modules
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#%%Load Data
# Load the data
def loadData():
    with np.load("/Users/matt/Documents/Winter_2019/ECE421/ECE421-a2/notMNIST.npz") as data:
        Data, Target = data["images"], data["labels"]
        np.random.seed(521)
        randIndx = np.arange(len(Data))
        np.random.shuffle(randIndx)
        Data = Data[randIndx] / 255.0
        Target = Target[randIndx]
        trainData, trainTarget = Data[:10000], Target[:10000]
        validData, validTarget = Data[10000:16000], Target[10000:16000]
        testData, testTarget = Data[16000:], Target[16000:]
    return trainData, validData, testData, trainTarget, validTarget, testTarget

trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()

def convertOneHot(trainTarget, validTarget, testTarget):
    newtrain = np.zeros((trainTarget.shape[0], 10))
    newvalid = np.zeros((validTarget.shape[0], 10))
    newtest = np.zeros((testTarget.shape[0], 10))

    for item in range(0, trainTarget.shape[0]):
        newtrain[item][trainTarget[item]] = 1
    for item in range(0, validTarget.shape[0]):
        newvalid[item][validTarget[item]] = 1
    for item in range(0, testTarget.shape[0]):
        newtest[item][testTarget[item]] = 1
    return newtrain, newvalid, newtest

new_trainTarget, new_validTarget, new_testTarget = convertOneHot(trainTarget, validTarget, testTarget)

def shuffle(trainData, trainTarget):
    np.random.seed(421)
    randIndx = np.arange(len(trainData))
    target = trainTarget
    np.random.shuffle(randIndx)
    data, target = trainData[randIndx], target[randIndx]
    return data, target

#%% Part 1 - Neural Network Using Numpy
# Implementation of a neural network using only Numpy - trained using gradient descent with momentum

def relu(x): # checked
    return x*(x>0)



def softmax(x): # checked

    return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True) 



def computeLayer(X, W, b): # checked
    return np.matmul(X,W) + b
    

def CE(target, prediction): # checked ; note: np.log() is ln()
#    print("CE: ")
#    print(-1*(np.sum(np.multiply(target,np.log(prediction))))/target.shape[0])
    return -1*(np.sum(np.multiply(target,np.log(prediction)))) \
        /target.shape[0]


# grad wrt output of softmax
#https://deepnotes.io/softmax-crossentropy
def gradCE(target, prediction):
#   -1/N * 1/Sk *tk
    return -1/(target.size)*np.sum(1/np.dot(prediction*target))








def compute_yhat(X,Wh,bh,Wo,bo):
    # forward pass
    z = computeLayer(X, Wh, bh) # 10000*1000
    h = relu(z) # 10000 * 1000
    s = computeLayer(h,Wo,bo) # 10000 * 10
    y_hat = softmax(s) # 10000 * 10
    
    return y_hat,h
    

def compute_accuracy(target, prediction):
    correct_count = 0
    for i in range(target.shape[0]):
        predicted_label = np.argmax(prediction[i])
        true_label = np.argmax(target[i])
        if predicted_label == true_label:
            correct_count += 1
        
        accuracy = correct_count / target.shape[0]
        
    return accuracy




    

#https://sudeepraja.github.io/Neural/
    
trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()

trainTarget, validTarget, testTarget = convertOneHot(trainTarget, validTarget, testTarget) # one hot encode
#input_layer_size = trainData.shape[1]*trainData.shape[2] # 28*28=784
input_layer_size = 784
trainData = (np.reshape(trainData,(trainData.shape[0],input_layer_size))) # resize
validData = (np.reshape(validData,(validData.shape[0],input_layer_size)))
testData = (np.reshape(testData,(testData.shape[0],input_layer_size)))

# init weight and bias
#input_layer_size = trainData.shape[1]*trainData.shape[2] # 28*28=784
hidden_layer_size = 1000 # given
output_layer_size = 10 # given
    
var_wh = 2/(input_layer_size + hidden_layer_size)
var_wo = 2/(hidden_layer_size + output_layer_size)
Wh = np.random.randn(input_layer_size, hidden_layer_size)*np.sqrt(var_wh) # 784 * 1000
Wo = np.random.randn(hidden_layer_size, output_layer_size)*np.sqrt(var_wo) # 1000 * 10
bh = np.full((1, hidden_layer_size), 0.1) #  1 * 1000 
bo = np.full((1, output_layer_size), 0.1) #  1 * 10


#z = X*Wh+bh  # 10000 * 1000
#h = relu(Z)  # 10000 * 1000
#s = U*Wo+bo  # 10000* 10
#y_hat = softmax(s) # 10000* 10


#%%

# forward prop
#X = trainData # 10000 * 28 * 28 #TODO: change to train
#Y = trainTarget # 10000 * 1
#X = (np.reshape(X,(X.shape[0],input_layer_size))) # resize to 10000 * 784
#z = computeLayer(X, Wh, bh) # 10000*1000
#h = relu(z) # 10000 * 1000
#s = computeLayer(h,Wo,bo) # 10000 * 10
#y_hat = softmax(s) # 10000 * 10
#CE_loss = CE(trainData, y_hat) # number



#delta_1 = (y_hat - Y).transpose() # 10 * 10000
#
#
#delta_2 = np.matmul(np.matmul(Wo,delta_1),np.sign(h)) # 1000 * 1000

#%%
#dJ_dWo = np.matmul(h,)



#
#dL_dWo = np.matmul(h.transpose(),delta_1.transpose()) # 1000 * 10 
#dL_dbo = delta_1.transpose() # 10000 * 10
#
##dL_dWh = np.matmul(X.transpose(),delta_2.transpose())
#dL_dWh = X@delta_2
#dL_dbh = delta_2.transpose() # 1000 * 1000

vh = np.ones((input_layer_size, hidden_layer_size))*1e-5 # 784 * 1000
vo = np.ones((hidden_layer_size, output_layer_size))*1e-5 # 1000 * 10
gamma = 0.9
alpha = 1e-4
train_loss, valid_loss, test_loss = ([] for i in range(3))
train_accuracy_list, valid_accuracy_list, test_accuracy_list = ([] for i in range(3))
X_train = trainData # 10000 * 28 * 28 #TODO: change to train
Y_train = trainTarget # 10000 * 
X_valid = validData
Y_valid = validTarget
X_test = testData 
Y_test = testTarget 


N = trainData.shape[0]

#y_hat_train = compute_yhat(X_train, Wh, bh, Wo, bo)
#y_hat_valid = compute_yhat(X_valid, Wh, bh, Wo, bo)
#y_hat_test = compute_yhat(X_test, Wh, bh, Wo, bo)
#
## backprop    
## delta_1 = dL/ds 
## grad of loss w.r.t. input to softmax 
#delta_1 = (y_hat_train - Y_train).transpose() # 10 * 10000
#
## delta_2 = dL/dz 
## grad of loss w.r.t. input to relu
#delta_2 = np.multiply((np.matmul(Wo,delta_1)).transpose(),np.sign(h)) # 10000 * 1000
#
#
#
#dL_dWh = X.transpose()@delta_2 # 784*1000
#dL_dbh = delta_2.transpose() # 1000 * 1000
#dL_dWo = np.matmul(h.transpose(),delta_1.transpose()) # 1000 * 10 
#dL_dbo = delta_1.transpose() # 100s00 * 10



#%%
#y_hat_train = compute_yhat(X_train, Wh, bh, Wo, bo)
#
#delta_1 = np.transpose(y_hat - Y_train)
for epoch in range(200):
    # forward pass
        # compute prediction
    y_hat_train, h_train = compute_yhat(X_train, Wh, bh, Wo, bo)
    y_hat_valid, h_valid= compute_yhat(X_valid, Wh, bh, Wo, bo)
    y_hat_test, h_test= compute_yhat(X_test, Wh, bh, Wo, bo)
        # compute loss
    CE_loss_train = CE(Y_train, y_hat_train) # number
    CE_loss_valid = CE(Y_valid, y_hat_valid) # number
    CE_loss_test = CE(Y_test, y_hat_test) # number

        # compute accuracy
    accuracy_train = compute_accuracy(Y_train, y_hat_train)
    accuracy_valid = compute_accuracy(Y_valid, y_hat_valid)
    accuracy_test = compute_accuracy(Y_test, y_hat_test)

        # append to lists for plotting
    train_loss.append(CE_loss_train)
    valid_loss.append(CE_loss_valid)
    test_loss.append(CE_loss_test)
    
    train_accuracy_list.append(accuracy_train)
    valid_accuracy_list.append(accuracy_valid)
    test_accuracy_list.append(accuracy_test)



    # backprop    
    # delta_1 = dL/ds 
    # grad of loss w.r.t. input to softmax 
    delta_1 = 1/N * (y_hat_train - Y_train).transpose() # 10 * 10000

    # delta_2 = dL/dz 
    # grad of loss w.r.t. input to relu
    delta_2 = np.multiply((np.matmul(Wo,delta_1)).transpose(),np.sign(h_train)) # 10000 * 1000
    
    
    
    dL_dWh = X_train.transpose()@delta_2 # 784*1000
    dL_dbh = delta_2.transpose() # 1000 * 1000
    dL_dWo = np.matmul(h_train.transpose(),delta_1.transpose()) # 1000 * 10 
    dL_dbo = delta_1.transpose() # 100s00 * 10

#    
    # update
    vh = gamma*vh + alpha*dL_dWh   # 784*1000
    Wh = Wh - vh
    vo = gamma*vo + alpha*dL_dWo ## 1000 * 10
    Wo = Wo - vo 
    
    print("training loss: {} validation loss: {} test loss: {}".format(CE_loss_train, CE_loss_valid, CE_loss_test))
    print("training acc: {} validation acc: {} test acc: {}".format(accuracy_train, accuracy_valid, accuracy_test))
#%%SGD Testing
#Run SGD algorithm

#Plot SGD results
x_axis = np.arange(200)+1

plt.figure(figsize=(10,10))

plt.subplot(211)
plt.plot(x_axis,CE_loss_train,color='c',linewidth=2.0,label="Training")
plt.plot(x_axis,CE_loss_valid,color='b',linewidth=2.0,label="Validation")
plt.plot(x_axis, CE_loss_test,color='g',linewidth=2.0,label="Test")
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.title("Neural Networks with" + hidden_layer_size+ "hidden units")
plt.legend()


plt.subplot(212)
plt.plot(x_axis,train_accuracy_list,color='c',linewidth=2.0,label="Training")
plt.plot(x_axis,valid_accuracy_list,color='b',linewidth=2.0,label="Validation")
plt.plot(x_axis, test_accuracy_list,color='g',linewidth=2.0,label="Test")
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend()

plt.show()    

#%% Part 2.1 - Convolutional Neural Network Model Implementation
def buildCNN(learningRate,dropoutRate,reg):
    #Reset BuildGraph
    tf.reset_default_graph()
    
    #CNN Parameters
    imageShape = [28,28]
    numFilters = 32
    filterShape = [3,3]
    maxpoolShape = [2,2]
    numInputChannels = 1
    numClasses = 10
    fc1_nodes = 784
    fc2_nodes = 10
    L2_regularizer = tf.contrib.layers.l2_regularizer(tf.constant(reg)) #reg = 0.0, turns off regularization
            
    #1. Input Layer
    X = tf.placeholder(tf.float32, shape=[None, imageShape[0],imageShape[1],1],name='X')
    Y = tf.placeholder(tf.float32, [None, numClasses],name='Y')  
    
    #2. 3x3 convolutional layer, with 32 filters, using vertical and horizontal strides of 1
    xavierInit = tf.contrib.layers.xavier_initializer()
    conv_filter = tf.get_variable(shape=[filterShape[0],filterShape[1],numInputChannels,numFilters], initializer=xavierInit,name="convFilter")
    conv_bias = tf.get_variable(shape=numFilters, initializer=xavierInit,name="convBias")
    conv_layer=tf.add(tf.nn.conv2d(X, conv_filter, strides=[1, 2, 2, 1], padding='SAME'),conv_bias)
    
    #3. ReLU Activation
    ReLu_output1 = tf.nn.relu(conv_layer)
    
    #4. Batch normalization layer
    batchMean , batchVar = tf.nn.moments(ReLu_output1,[0,1,2]) 
    batchNorm_layer = tf.nn.batch_normalization(ReLu_output1,batchMean,batchVar,offset = None, scale = None,variance_epsilon=1e-3)
    
    #5. Max 2×2 max pooling layer
    maxPool_layer = tf.nn.max_pool(batchNorm_layer,ksize =[1,maxpoolShape[0],maxpoolShape[1],1],strides = [1,1,1,1],padding = 'SAME')
    
    #6. Flatten layer
    numFeatures = int(imageShape[0]/maxpoolShape[0]*imageShape[1]/maxpoolShape[1]*numFilters)
    flatten_layer = tf.reshape(maxPool_layer,[-1,numFeatures])
    
    #7. Fully connected layer (with 784 output units, i.e. corresponding to each pixel) with dropout if need be
    W_fc1 = tf.get_variable(shape=[numFeatures,fc1_nodes],initializer = xavierInit ,regularizer =  L2_regularizer, name ="FC1_Weight")
    b_fc1 = tf.get_variable(shape=[fc1_nodes],initializer = xavierInit, name="FC1_bias")
    temp_fc_layer1 = tf.add(tf.matmul(flatten_layer,W_fc1),b_fc1)
    
    dropoutFlag = tf.placeholder(tf.bool,name='dropoutFlag')
    keepProb = tf.constant(1-dropoutRate,dtype='float32')
    fc_layer1 = tf.cond(dropoutFlag,lambda: tf.nn.dropout(temp_fc_layer1,keepProb),lambda: temp_fc_layer1)
    
    #8. ReLU Activation 2
    ReLu_output2 = tf.nn.relu(fc_layer1)
    
    #9. Fully connected layer (with 10 output units, i.e. corresponding to each class)
    W_fc2 = tf.get_variable(shape=[fc1_nodes,fc2_nodes],initializer = xavierInit,regularizer =  L2_regularizer, name = "FC2_Weights")
    b_fc2 = tf.get_variable(shape=[fc2_nodes],initializer = xavierInit, name = "FC2_bias")
    fc_layer2 = tf.add(tf.matmul(ReLu_output2,W_fc2),b_fc2)
    
    #10. Softmax output
    pred = tf.nn.softmax(fc_layer2)
    
    #11. Cross Entropy loss
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=Y,logits=fc_layer2))
    
    #Optimizer
    optimizer = tf.train.AdamOptimizer(learningRate,name = "ADAM").minimize(loss)
    
    #Accuracy
    equalityScore = tf.equal(tf.cast(tf.arg_max(pred,1),tf.float32),tf.cast(tf.arg_max(Y,1),tf.float32))
    accuracy = tf.reduce_mean(tf.cast(equalityScore,tf.float32))
    
    return X, Y, dropoutFlag, optimizer, loss, accuracy

#%% Part 2.2 - Convolutional Neural Network Model Training
def trainCNN(learningRate,iterations,batchSize,reg,dropout):
    #Load and Reshape Data
    trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()
    trainTarget, validTarget, testTarget = convertOneHot(trainTarget, validTarget, testTarget)
    
    N_trainData = trainData.shape[0]
    N_validData = validData.shape[0]
    N_testData = testData.shape[0]
    
    trainData = np.reshape(trainData,(N_trainData,784))
    validData = np.reshape(validData,(N_validData,784))
    testData = np.reshape(testData,(N_testData,784))
    
    trainData = trainData.reshape(-1, 28, 28, 1)
    testData = testData.reshape(-1, 28, 28, 1)
    validData = validData.reshape(-1, 28, 28, 1)
    
    #Build CNN
    X,Y,dropoutFlag, optimizer, loss, accuracy=buildCNN(learningRate,dropout,reg)
    
    #Initialize CNN Loss and Accuracy Arrays
    trainLoss,trainAcc,validLoss,validAcc,testLoss,testAcc = ([] for i in range(6))
    
    #Start Training Session
    with tf.Session() as sess:
        #Run the initializer
        init = tf.global_variables_initializer()
        sess.run(init)
        
        for epoch in range(iterations):
            #Calculate # of Batches
            batchIterations = len(trainData)//batchSize
            
            #Train model
            for i in range(batchIterations):
                #Create batch
                xBatch = trainData[i*batchSize:min((i+1)*batchSize,len(trainData))]
                yBatch = trainTarget[i*batchSize:min((i+1)*batchSize,len(trainTarget))]
                
                #SGD using current batch
                optBatch = sess.run(optimizer, feed_dict={X: xBatch,Y: yBatch,dropoutFlag: 1})
            
            #Calculate loss and accuracy
            current_trainLoss,current_trainAcc = sess.run([loss,accuracy], feed_dict={X: trainData,Y: trainTarget,dropoutFlag: 0})
            trainLoss.append(current_trainLoss)
            trainAcc.append(current_trainAcc)
            print("Epoch: {0}, train loss: {1:.2f}, train accuracy: {2:.01%}". format(epoch + 1, current_trainLoss, current_trainAcc))
            
            current_validLoss,current_validAcc = sess.run([loss,accuracy], feed_dict={X: validData,Y: validTarget,dropoutFlag: 0})
            validLoss.append(current_validLoss)
            validAcc.append(current_validAcc)
            
            current_testLoss,current_testAcc = sess.run([loss,accuracy], feed_dict={X: testData,Y: testTarget,dropoutFlag: 0})
            testLoss.append(current_testLoss)
            testAcc.append(current_testAcc)
            
            #Shuffle Training Dataset and Labels
            trainData, trainTarget = shuffle(trainData,trainTarget)
        
        print("Optimization finished!")
        return trainLoss, validLoss, testLoss, trainAcc, validAcc, testAcc

#%% Part 2.2/2.3 - Convolutional Neural Network Training and HyperParameter Investigation
#Set Model Parameters
learningRate = 0.0001 #1e-4
iterations = 50
batchSize = 32
Lambda = 0.0
dropout = 0.5

CNN_trainLoss,CNN_validLoss, CNN_testLoss, CNN_trainAcc, CNN_validAcc, CNN_testAcc= trainCNN(learningRate,iterations,batchSize,Lambda,dropout)

#Plot Results
x_axis = np.arange(iterations)+1
    
plt.figure(figsize=(10,10))
    
plt.subplot(211)
plt.plot(x_axis,CNN_trainLoss,color='c',linewidth=2.0,label="Training")
plt.plot(x_axis,CNN_validLoss,color='b',linewidth=2.0,label="Validation")
plt.plot(x_axis, CNN_testLoss,color='g',linewidth=2.0,label="Test")
plt.ylabel('Loss')
plt.xlabel('Epochs')
if Lambda!=0:
    plt.title("Batch Stoichastic Gradient Descent with Lambda = "+str(Lambda))
if dropout!=0:
    plt.title("Batch Stoichastic Gradient Descent with Dropout = "+str(dropout))
plt.legend()

    
plt.subplot(212)
plt.plot(x_axis,CNN_trainAcc,color='c',linewidth=2.0,label="Training")
plt.plot(x_axis,CNN_validAcc,color='b',linewidth=2.0,label="Validation")
plt.plot(x_axis,CNN_testAcc,color='g',linewidth=2.0,label="Test")
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend()
    
plt.show()
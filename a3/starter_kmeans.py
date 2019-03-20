#Import Modules
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import helper as hlp

# Loading data
data = np.load('data2D.npy')
#data = np.load('data100D.npy')
[num_pts, dim] = np.shape(data)
is_valid = 1

# For Validation set
if is_valid:
  valid_batch = int(num_pts / 3.0)
  np.random.seed(45689)
  rnd_idx = np.arange(num_pts)
  np.random.shuffle(rnd_idx)
  val_data = data[rnd_idx[:valid_batch]]
  data = data[rnd_idx[valid_batch:]]

# Distance function for K-means
def distanceFunc(X, MU):
    # Inputs - X: is an NxD matrix (N observations and D dimensions), MU: is an KxD matrix (K means and D dimensions)
    # Outputs - pair_dist: is the pairwise distance matrix (NxK)
    
    X_squared = tf.reduce_sum(tf.square(X),axis=1,keepdims=True) #Nx1
    MU_squared = tf.reduce_sum(tf.square(MU),axis=1) #Kx1
    X_MU = tf.matmul(X,MU,transpose_b=True) #NxK matrix
    
    pair_dist = X_squared-2*X_MU+MU_squared
    
    return pair_dist

#Model Parameters
K = 5 #Number of clusters
N = num_pts-valid_batch #Number of training data points
D = dim # Dimension of data
MAX_ITERS = 1000
LEARNING_RATE = 0.01
np.random.seed(421) 

#Build K-means Graph
#Reset to defaultgraph
tf.reset_default_graph()

if D==2:
    points = tf.placeholder(dtype=tf.float64, shape=[None,D],name='points')
    centroid_init = tf.truncated_normal(shape=[K,D],dtype = tf.float64)
    centroids = tf.get_variable(dtype = tf.float64,initializer = centroid_init, name = "centroids")
elif D==100:
    point = tf.placeholder(tf.float32,shape=[None,D],name='points')
    centroid_init = tf.truncated_normal(shape=[K,D],dtype = tf.float32)
    centroids = tf.get_variable(dtype = tf.float32,initializer = centroid_init, name = "centroids")


distances = distanceFunc(points,centroids)
assignment = tf.argmin(distances,axis=1)

loss = tf.reduce_sum(tf.reduce_min(distances,axis=1,keepdims=True))

optimizer = tf.train.AdamOptimizer(LEARNING_RATE,beta1=0.9,beta2=0.99,epsilon=1e-5).minimize(loss)

#Train and Test K-means Model
with tf.Session() as sess:
    #Run the initializer
    init = tf.global_variables_initializer()
    sess.run(init) 
    trainLoss = []
    
    #Train Model
    for epoch in range(MAX_ITERS):                       
        opt,current_trainLoss = sess.run([optimizer,loss], feed_dict={points:data})
        trainLoss.append(current_trainLoss)

    print("Optimization finished!")
    
    #Test Model
    trainAssignment,trainCentroids = sess.run([assignment,centroids],feed_dict={points:data})
    validAssignment,validLoss = sess.run([assignment,loss],feed_dict={points:val_data})
    print("Validation Loss: " +str(validLoss))
    clusterPct = np.zeros(K)
    for i in range(valid_batch):
        clusterPct[validAssignment[i]]+=1
    clusterPct*=100/valid_batch
    for i in range(K):
        print("Cluster"+str(i+1)+": "+"{0:.2f}".format(round(clusterPct[i],2))+"%")

#Print out plot
if D==2:
    plt.figure(figsize=(10,10)) #figsize=(width,height)

    #Plot 1 - 2D Scatter
    plt.subplot(211)
    c_assignment = np.asarray(trainAssignment,dtype=np.float32).reshape(np.shape(data[:,0]))
    plt.scatter(data[:,0],data[:,1],c=c_assignment,cmap='jet',marker='.')
    plt.scatter(trainCentroids[:,0],trainCentroids[:,1],c='k',marker = 'o')
    plt.title("Kmeans Learning with K = "+str(K))
    plt.ylabel('x1')
    plt.xlabel('x2')
    

    #Plot 2 - Loss
    plt.subplot(212)
    x_axis = np.arange(MAX_ITERS)+1
    plt.plot(x_axis,trainLoss,color='c',linewidth=2.0,label="Training")
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend()

else:
    #Plot 1 - Loss
    plt.figure(figsize=(10,5)) #figsize=(width,height)
    x_axis = np.arange(MAX_ITERS)+1
    plt.plot(x_axis,trainLoss,color='c',linewidth=2.0,label="Training")
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.title("Kmeans Learning with K = "+str(K))
    plt.legend()
    
    
plt.show()
    



import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import helper as hlp

# Loading data
data = np.load('data100D.npy')
#data = np.load('data2D.npy')
[num_pts, dim] = np.shape(data)

is_valid = 1
run_kmeans = 1 # set to 1 if want to run both kmeans and MoG

# For Validation set
if is_valid:
  valid_batch = int(num_pts / 3.0)
  np.random.seed(45689) 
  rnd_idx = np.arange(num_pts)
  np.random.shuffle(rnd_idx)
  val_data = data[rnd_idx[:valid_batch]]
  data = data[rnd_idx[valid_batch:]]

# Distance function for GMM
def distanceFunc(X, MU):
    # Inputs
    # X: is an NxD matrix (N observations and D dimensions)
    # MU: is an KxD matrix (K means and D dimensions)
    # Outputs
    # pair_dist: is the pairwise distance matrix (NxK)
    # TODO


    X = tf.cast(X,tf.float32)
    X_squared = tf.reduce_sum(tf.square(X),axis=1,keepdims=True) #Nx1
    MU_squared = tf.reduce_sum(tf.square(MU),axis=1) #Kx1
    X_MU = tf.matmul(X,MU,transpose_b=True) #NxK matrix
    
    pair_dist = X_squared-2*X_MU+MU_squared
    
    return pair_dist


def log_GaussPDF(X, mu, sigma):  #logP(X|z)
    # Inputs
    # X: N X D
    # mu: K X D
    # sigma: K X 1

    # Outputs:
    # log Gaussian PDF N X K
    
    D = X.shape[1]    
    sigma = tf.transpose(sigma) # 1xK
    distance = distanceFunc(X,mu) # NxK
#    print(distance.get_shape())
    
    part1 =  -1* (tf.divide(distance, 2*tf.square(sigma))) # NxK
    part2 =  -1* (tf.multiply(D/2,tf.log(2*np.pi*tf.square(sigma)))) # 1xK

    logpdf = tf.add(part1, part2) 
#    print(logpdf.get_shape())
    
    return logpdf
#%%
def log_posterior(log_PDF, log_pi):  #logP(z|X)
    # Input
    # log_PDF: log Gaussian PDF N X K
    # log_pi: K X 1
    
    # Outputs
    # log_post: N X K
    
    # TODO
    N = log_PDF.shape[0]
    part1 = log_PDF + log_pi # NxK
#    print(part1.get_shape)
    part2 = hlp.reduce_logsumexp(log_PDF + log_pi, reduction_indices=1) # Nx
    part2 = tf.reshape(part2,(N,1)) # Nx1
#    print(part2.get_shape)
    log_pst = part1 -part2 # NxK
#    print(log_pst.get_shape)
    return log_pst


#%% build graph and train model

    

# build graph    
#Reset to defaultgraph
tf.reset_default_graph()
D = data.shape[1]
K = 15 #Number of clusters
LEARNING_RATE = 0.01
MAX_ITERS = 1000



points = tf.placeholder(dtype=tf.float32, shape=[None,D],name='points')
centroid_init = tf.truncated_normal(shape=[K,D],dtype = tf.float32)
sigma_init = tf.truncated_normal(shape=[K,1],dtype = tf.float32)
centroids = tf.get_variable(dtype = tf.float32,initializer = centroid_init, name = "centroids")

distances = distanceFunc(points,centroids) # for kmeans only

log_pi = hlp.logsoftmax(tf.Variable(tf.random_normal([K,1],dtype = tf.float32)))

phi = tf.Variable(tf.truncated_normal(shape=[K,1],dtype = tf.float32))
good_sigma_square = tf.exp(phi)
good_sigma = tf.sqrt(good_sigma_square)
good_pi = hlp.logsoftmax(good_sigma_square)


log_pdf = log_GaussPDF(data,centroids,good_sigma)
log_pi = tf.transpose(good_pi)
log_pst = log_posterior(log_pdf, log_pi)

log_loss = hlp.reduce_logsumexp(log_pdf + log_pi,reduction_indices=1)

# loss and optimizer for MoG

loss = tf.reduce_sum(-1*log_loss) # negative log loss
optimizer = tf.train.AdamOptimizer(LEARNING_RATE, beta1=0.9,beta2=0.99,epsilon=1e-5).minimize(loss)

assignment = tf.argmax(log_pdf,axis=1) # Nx (10000,)
print("assignment shape:"+str(assignment.get_shape()))


# loss for kmeans 
loss_kmeans = tf.reduce_sum(tf.reduce_min(distances,axis=1,keepdims=True))
optimizer_kmeans = tf.train.AdamOptimizer(LEARNING_RATE,beta1=0.9,beta2=0.99,epsilon=1e-5).minimize(loss)
assignment_kmeans = tf.argmin(distances,axis=1)

with tf.Session() as sess:
    #Run the initializer
    init = tf.global_variables_initializer()
    sess.run(init) 
    trainLoss = []
    trainLoss_kmeans = []
    
    # train model
        #Train Model
    for epoch in range(MAX_ITERS):                       
        
        # MoG
        opt,current_trainLoss = sess.run([optimizer,loss], feed_dict={points:data})
        trainLoss.append(current_trainLoss)
        print("TrainLoss:"+str(current_trainLoss))
        
        # kmeans
        if run_kmeans ==1:
            opt_kmeans,current_trainLoss_kmeans = sess.run([optimizer_kmeans,loss_kmeans], feed_dict={points:data})
            trainLoss_kmeans.append(current_trainLoss_kmeans)
        
    print(centroids.eval()) # print assigned centroid
    print(good_sigma.eval()) # print final sigma
    print("Optimization finished!")
    
    #Test Model
    # Assign label to dataset
    trainAssignment = sess.run([assignment],feed_dict={points:data})

    # if has valid data
    if (is_valid ==1): 
        #MoG
        validAssignment,validLoss = sess.run([assignment,loss],feed_dict={points:val_data})
        print("Validation Loss MoG: " +str(validLoss))
        clusterPct = np.zeros(K)
        for i in range(valid_batch):
            clusterPct[validAssignment[i]]+=1
        clusterPct/= valid_batch
        for i in range(K):
            print("Cluster"+str(i+1)+": "+str(clusterPct[i]))
            
        #kmeans
        if run_kmeans ==1:
            validAssignment_kmeans,validLoss_kmeans = sess.run([assignment_kmeans,loss_kmeans],feed_dict={points:val_data})
            print("Validation Loss Kmeans: " +str(validLoss_kmeans))
            clusterPct_kmeans = np.zeros(K)
            for i in range(valid_batch):
                clusterPct_kmeans[validAssignment_kmeans[i]]+=1
            clusterPct_kmeans/= valid_batch
            for i in range(K):
                print("Cluster kmeans"+str(i+1)+": "+str(clusterPct_kmeans[i]))

            
#%%
#Print out plot
if D==2:
    plt.figure(figsize=(10,10)) #figsize=(width,height)

    #Plot 1 - 2D Scatter for training
    
    if is_valid == 1:        # only plot on valid data if has valid data
        plt.subplot(211)
        c_assignment = np.asarray(validAssignment,dtype=np.float32).reshape(np.shape(data[:,0]))
        plt.scatter(data[:,0],data[:,1],c=c_assignment,cmap='jet')
        plt.title("Kmeans with K = "+str(K))
        plt.ylabel('x1')
        plt.xlabel('x2')
        
        
    else: # plot training data
        plt.subplot(211)
        c_assignment = np.asarray(trainAssignment,dtype=np.float32).reshape(np.shape(data[:,0]))
        plt.scatter(data[:,0],data[:,1],c=c_assignment,cmap='jet')
        plt.title("Kmeans with K = "+str(K))
        plt.ylabel('x1')
        plt.xlabel('x2')
    

    #Plot 2 - Loss
    plt.subplot(212)
    x_axis = np.arange(MAX_ITERS)+1
    plt.plot(x_axis,trainLoss,color='c',linewidth=2.0,label="Training-MoG")
    if run_kmeans == 1:
            plt.plot(x_axis,trainLoss_kmeans,color='r',linewidth=2.0,label="Training-Kmeans")

    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend()

else:
    #Plot 1 - Loss
    plt.figure(figsize=(10,5)) #figsize=(width,height)
    x_axis = np.arange(MAX_ITERS)+1
    
    plt.plot(x_axis,trainLoss,color='c',linewidth=2.0,label="Training-MoG")
    if run_kmeans == 1:
        plt.plot(x_axis,trainLoss_kmeans,color='r',linewidth=2.0,label="Training-Kmeans")
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.title("Kmeans and MoG with K = "+str(K))
    plt.legend()
    
    
plt.show()
    
#    print(sess.run(log_pdf.get_shape()))
    
    



 
#%%


    
    
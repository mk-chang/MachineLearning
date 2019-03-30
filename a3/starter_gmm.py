import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import helper as hlp

# Loading data
#data = np.load('data100D.npy')
data = np.load('data2D.npy')
[num_pts, dim] = np.shape(data)

# For Validation set
#if is_valid:
#  valid_batch = int(num_pts / 3.0)
#  np.random.seed(45689) 
#  rnd_idx = np.arange(num_pts)
#  np.random.shuffle(rnd_idx)
#  val_data = data[rnd_idx[:valid_batch]]
#  data = data[rnd_idx[valid_batch:]]

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
#    D = 100
    sigma = tf.transpose(sigma) # 1xK
    distance = distanceFunc(X,mu) # NxK
#    print(distance.get_shape())
    
    part1 =  tf.divide(distance, 2*tf.square(sigma)) # NxK
    part2 =  tf.multiply(D/2,tf.log(2*np.pi*tf.square(sigma))) # 1xK

    logpdf = tf.subtract(part2, part1) 
    
    
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





#Reset to defaultgraph
tf.reset_default_graph()
D = data.shape[1]
K = 3 #Number of clusters

if D==2:
    points = tf.placeholder(dtype=tf.float32, shape=[None,D],name='points')
    centroid_init = tf.truncated_normal(shape=[K,D],dtype = tf.float32)
    sigma_init = tf.truncated_normal(shape=[K,1],dtype = tf.float32)
    centroids = tf.get_variable(dtype = tf.float32,initializer = centroid_init, name = "centroids")
    log_pi = hlp.logsoftmax(tf.Variable(tf.random_normal([K,1],dtype = tf.float32)))
elif D==100:
    point = tf.placeholder(tf.float32,shape=[None,D],name='points')
    centroid_init = tf.truncated_normal(shape=[K,D],dtype = tf.float32)
    centroids = tf.get_variable(dtype = tf.float32,initializer = centroid_init, name = "centroids")
    log_pi = hlp.logsoftmax(tf.Variable(tf.random_normal([K,1],dtype = tf.float32)))





log_pdf = log_GaussPDF(data,centroid_init,sigma_init)
log_pi = tf.transpose(log_pi)
log_pst = log_posterior(log_pdf, log_pi)
print("logpdf shape"+str(log_pdf.get_shape()))
print("log_pi shape"+str(log_pi.get_shape()))
print("log_pst shape"+str(log_pst.get_shape()))


with tf.Session() as sess:
    #Run the initializer
    init = tf.global_variables_initializer()
    sess.run(init) 
    
#    print(sess.run(log_pdf.get_shape()))
    
    



 
#%%


    
    
import cv2
import xlrd;
import numpy as np;
import matplotlib.pyplot as plt

import matplotlib.image as mpimg
from sklearn.cluster import DBSCAN, SpectralClustering
from sklearn import cluster, datasets, mixture
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn_extensions.fuzzy_kmeans import FuzzyKMeans
from PIL import Image
image_size=np.zeros([254,2]);
def median_filter(data, filter_size):
    temp = []
    indexer = filter_size // 2
    data_final = []
    data_final = np.zeros((len(data),len(data[0])))
    for i in range(len(data)):

        for j in range(len(data[0])):

            for z in range(filter_size):
                if i + z - indexer < 0 or i + z - indexer > len(data) - 1:
                    for c in range(filter_size):
                        temp.append(0)
                else:
                    if j + z - indexer < 0 or j + indexer > len(data[0]) - 1:
                        temp.append(0)
                    else:
                        for k in range(filter_size):
                            temp.append(data[i + z - indexer][j + k - indexer])

            temp.sort()
            data_final[i][j] = temp[len(temp) // 2]
            temp = []
    return data_final

filter_arr = []

#file_path="/Volumes/uok/brain tumor/brain-mri-images-for-brain-tumor-detection/yes/";
i=11

img_filter1 = Image.open(str(i) + ".jpg").convert('L')
arr = np.asarray(img_filter1)

removed_noise = median_filter(arr, 3) 
arr = Image.fromarray(removed_noise)

plt.imshow(arr, cmap='gray', vmin=0, vmax=255)
plt.show()

size_i=np.shape(arr)
data=(np.array(arr)).flatten()
data = np.reshape(data, (-1, 1))


import numpy as np
from sklearn.datasets.samples_generator import make_circles, make_blobs, make_swiss_roll, make_moons, make_s_curve
from matplotlib import pyplot as plt
from scipy.linalg import fractional_matrix_power
import kmean_code;


def dist_sqrt(X1, X2):
    s=np.shape(X1);
    sm=0;
    for i in range(s[0]):
        sm=sm+(X1[i]-X2[i])*(X1[i]-X2[i])   #Euclidean distance
    return np.sqrt(sm); 

def fun_dist_matrix(data):
    s1=np.shape(data)
    dist_matrix=np.zeros([s1[0],s1[0]]);
    for i in range(s1[0]):
        for j in range(s1[0]):
            dist_matrix[i, j]=dist_sqrt(data[i],data[j])
    return dist_matrix;        
    
#Fully-connected symmetry favored similarity graph using Gaussian similarity function    
def pair_distances(dist_matrix, nearest_neighbor):    
    s1, s2=np.shape(dist_matrix);
    A=np.zeros([s1, s2]);   #affinity matrix
    dist_c=np.zeros([s1]);  #c-nearest neighbor distance 
    W_c=np.zeros([s1,s2]);   #column-wise c-nearest neighbors 0 or 1
    

    # column-wise c-nearest neighbors
    for i in range(s1):
        dist_c[i]=sorted(dist_matrix[:,i])[nearest_neighbor];
        for j in range(s2):
            if dist_matrix[i,j]<=dist_c[i]:        #column-wise c-nearest neighbors
                W_c[j,i]=1;
            else:
                W_c[j,i]=0;
    
    def affinity(i1,i2):
        if (W_c[i2, i1]==1):
            return np.exp(-(dist_matrix[i2, i1]*dist_matrix[i2, i1])/(dist_c[i1]*dist_c[i2]))
        else:
            return 0
        
        
    # affinity matrix
    for i in range(s1):
        for j in range(s1):
            if (W_c[i,j]==1 and W_c[j,i]==1):
                A[i,j]=1;
            if (W_c[i,j]==1 and (W_c[j,i] != 1)):
                A[i,j]=affinity(j,i);
            else:
                A[i,j]=affinity(i,j);
    return A
    
    
    

def spectral_code(X, no_clusters, nearest_neighbor, samples):    
    #distance matrix
    dist_matrix=fun_dist_matrix(X)
    
    #Fully-connected (Affinity matrix) similarity graph using Gaussian similarity function 
    W = pair_distances(dist_matrix, nearest_neighbor)
    
    
    # degree matrix
    D = np.diag(np.sum(W, axis=1))
    
    # laplacian matrix
    L = D - W
    Lno=np.dot(fractional_matrix_power(D, -0.5), np.dot(L, fractional_matrix_power(D, -0.5)))
    
    #column v[:,i] is the eigenvector corresponding to the eigenvalue e[i]
    
    e, v = np.linalg.eig(Lno) 
    e_indx=sorted(range(len(e)), key=lambda i: e[i], reverse=False)[:no_clusters];
    s1, s2=np.shape(X);
    eigenvector_space=np.zeros([s1, no_clusters]);
    j=0;
    for i in e_indx:
        eigenvector_space[:,j]=v[:,i];
        j=j+1;
    
    
    #km = KMeans(n_clusters=no_clusters, init='k-means++')
    #km_clustering=km.fit(eigenvector_space)
    
    labels_=kmean_code.kmean_clus_code(no_clusters, s1, no_clusters, eigenvector_space)
    return labels_


Y_p = spectral_code(data[0:1000]+1,2,3,22605)
Y_p=np.resize(Y_p, np.size(Y_p)).reshape(size_i[0],size_i[1])
plt.imshow(Y_p, cmap='gray', vmin=0, vmax=1)
plt.show()
#Data for example
#X = np.array([
#    [1, 3], [2, 1], [1, 1],
#    [3, 2], [7, 8], [9, 8],
#    [9, 9], [8, 7], [13, 14],
#    [14, 14], [15, 16], [14, 15]
#])

#no_clusters=2 
#nearest_neighbor=3 
#samples=500
##X, clusters = make_circles(n_samples=samples, noise=.05, factor=.5, random_state=0)
##X, clusters = make_blobs(n_samples=samples)
#X, clusters = make_moons(n_samples=samples)
#fig = plt.figure(figsize=[6, 6])
#plt.scatter(X[:,0], X[:,1], c=clusters, cmap='rainbow', alpha=0.7, edgecolors='b')
#
#labels_=spectral_code(X, no_clusters, nearest_neighbor, samples)
#
#fig = plt.figure(figsize=[6, 6])
#
#plt.scatter(X[:,0], X[:,1], c=labels_, cmap='rainbow', alpha=0.7, edgecolors='b')


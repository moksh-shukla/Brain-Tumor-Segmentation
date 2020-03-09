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

from copy import deepcopy
import pandas as pd
from sklearn.datasets.samples_generator import make_circles, make_blobs, make_swiss_roll, make_moons, make_s_curve
from scipy.misc import imshow

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


#i=12

img_filter1 = Image.open("N211.jpg").convert('L')
arr = np.asarray(img_filter1)

removed_noise = median_filter(arr, 3) 
arr = Image.fromarray(removed_noise)

plt.imshow(arr, cmap='gray', vmin=0, vmax=255)
plt.show()

size_i=np.shape(arr)
data=(np.array(arr)).flatten()
data = np.reshape(data, (-1, 1))

# Euclidean Distance Caculator
def dist_sqrt(X1, X2):
    s=np.shape(X1);
    sm=0;
    for i in range(s[0]):
        sm=sm+(X1[i]-X2[i])*(X1[i]-X2[i])   #Euclidean distance
    return np.sqrt(sm); 

def fun_dist_matrix(center, data):
    s1=np.shape(data)
    s2=np.shape(center)
    dist_matrix=np.zeros([s1[0],s2[0]]);
    for i in range(s1[0]):
        for j in range(s2[0]):
            dist_matrix[i, j]=dist_sqrt(data[i],center[j])
    return dist_matrix;
    
def fun_assign_label(dist_matrix):
    s1=np.shape(dist_matrix)
    class_label=np.zeros([s1[0]])
    for i in range(s1[0]):
        class_label[i]=np.argmin(dist_matrix[i,:])
    return class_label

def fun_center(data, class_label, k):
    s1=np.shape(data)
    center=np.zeros([k, s1[1]])
    center_count_element=np.zeros([k])
    for i in range(s1[0]):
        for j in range(s1[1]):
            center[class_label[i], j]=center[class_label[i], j]+data[i, j]
        center_count_element[class_label[i]]=center_count_element[class_label[i]]+1
    for i in range(k):
        for j in range(s1[1]):
            center[i,j]=center[i,j]/center_count_element[i]
    return center
    
def func_total_cost(data, class_label, dist_matrix):
    s1=np.shape(data)
    total_cost=0
    for i in range(s1[0]):
        total_cost=total_cost+dist_matrix[i, class_label[i]]
    return total_cost

#def kmean_clus_code(k, samples, no_features, data, cluster_indx): #optinal
def kmean_clus_code(k, samples, no_features, data):
#    k=3   # number of clusters
#    samples=500   # number of samples
#    no_features=2
#    
#    #data generation
#    data, cluster_indx=make_blobs(samples)
    
    #center point
    center=np.zeros([k,no_features]);
    
    # Cluster Lables
    #cluster_label = np.zeros([samples])
    
#    plt.figure();
#    
#    # Getting the values and plotting it
#    
#    plt.scatter(data[:,0], data[:,1], c=cluster_indx, cmap='rainbow', alpha=0.7, edgecolors='b')
#    
    #random initialization of center
    # Plotting along with the Centroids
    for i in range(k):
        temp=np.random.randint(0, samples-1);
        center[i]=data[temp]
    
#    plt.scatter(center[:,0], center[:,1], marker='*', s=200, c='k')
    
    
    dist_matrix=fun_dist_matrix(center, data);
    class_label=fun_assign_label(dist_matrix);
    class_label=class_label.astype(np.int64)
    cost1=func_total_cost(data, class_label, dist_matrix);
    
    center=fun_center(data, class_label, k)
    
    error=1;
    
    while(error>0):
        dist_matrix=fun_dist_matrix(center, data);
        class_label=fun_assign_label(dist_matrix);
        class_label=class_label.astype(np.int64)
        cost2=func_total_cost(data, class_label, dist_matrix);
        center=fun_center(data, class_label, k)
        error=cost2-cost1;
        cost1=cost2
    
        
#    plt.figure();
#    
#    # Getting the values and plotting it
#    
#    plt.scatter(data[:,0], data[:,1], c=class_label, cmap='rainbow', alpha=0.7, edgecolors='b')
#    
#    
#    plt.scatter(center[:,0], center[:,1], marker='*', s=200, c='k')
    return class_label;

Y_p = kmean_clus_code(2,52734,1,data)
Y_p=np.resize(Y_p, np.size(Y_p)).reshape(size_i[0],size_i[1])
plt.imshow(Y_p, cmap='gray', vmin=0, vmax=1)

plt.show()


#from scipy.misc import toimage
#toimage(Y_p).show()



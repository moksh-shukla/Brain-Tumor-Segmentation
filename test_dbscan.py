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
i=110

img_filter1 = Image.open(str(i) + ".jpg").convert('L')
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
    s=numpy.shape(X1);
    sm=0;
    for i in range(s[0]):
        sm=sm+(X1[i]-X2[i])*(X1[i]-X2[i])   #Euclidean distance
    return numpy.sqrt(sm); 

def fun_dist_matrix(data):
    s1=numpy.shape(data)
    dist_matrix=numpy.zeros([s1[0],s1[0]]);
    for i in range(s1[0]):
        for j in range(s1[0]):
            dist_matrix[i, j]=dist_sqrt(data[i],data[j])
    return dist_matrix;

def set2List(NumpyArray):
    list = []
    for item in NumpyArray:
        list.append(item.tolist())
    return list


def GenerateData():
    x1=np.random.randn(50,2)
    x2x=np.random.randn(80,1)+12
    x2y=np.random.randn(80,1)
    x2=np.column_stack((x2x,x2y))
    x3=np.random.randn(100,2)+8
    x4=np.random.randn(120,2)+15
    z=np.concatenate((x1,x2,x3,x4))
    return z


def DBSCAN(Dataset, Epsilon,MinumumPoints):
#    Dataset is a mxn matrix, m is number of item and n is the dimension of data
    m,n=Dataset.shape
    Visited=np.zeros(m,'int')
    Type=np.zeros(m)
#   -1 noise, outlier
#    0 border
#    1 core
    ClustersList=[]
    Cluster=[]
    PointClusterNumber=np.zeros(m)
    PointClusterNumberIndex=1
    PointNeighbors=[]
    DistanceMatrix = fun_dist_matrix(Dataset)
    for i in range(m):
        if Visited[i]==0:
            Visited[i]=1
            PointNeighbors=np.where(DistanceMatrix[i]<Epsilon)[0]
            if len(PointNeighbors)<MinumumPoints:
                Type[i]=-1
            else:
                for k in range(len(Cluster)):
                    Cluster.pop()
                Cluster.append(i)
                PointClusterNumber[i]=PointClusterNumberIndex
                
                
                PointNeighbors=set2List(PointNeighbors)    
                ExpandClsuter(Dataset[i], PointNeighbors,Cluster,MinumumPoints,Epsilon,Visited,DistanceMatrix,PointClusterNumber,PointClusterNumberIndex  )
                Cluster.append(PointNeighbors[:])
                ClustersList.append(Cluster[:])
                PointClusterNumberIndex=PointClusterNumberIndex+1
                 
                    
    return PointClusterNumber 



def ExpandClsuter(PointToExapnd, PointNeighbors,Cluster,MinumumPoints,Epsilon,Visited,DistanceMatrix,PointClusterNumber,PointClusterNumberIndex  ):
    Neighbors=[]

    for i in PointNeighbors:
        if Visited[i]==0:
            Visited[i]=1
            Neighbors=np.where(DistanceMatrix[i]<Epsilon)[0]
            if len(Neighbors)>=MinumumPoints:
#                Neighbors merge with PointNeighbors
                for j in Neighbors:
                    try:
                        PointNeighbors.index(j)
                    except ValueError:
                        PointNeighbors.append(j)
                    
        if PointClusterNumber[i]==0:
            Cluster.append(i)
            PointClusterNumber[i]=PointClusterNumberIndex
    return


Y_p = DBSCAN(data,3.2,390)
Y_p=np.resize(Y_p, np.size(Y_p)).reshape(size_i[0],size_i[1])
plt.imshow(Y_p, cmap='gray', vmin=0, vmax=1)
plt.show()

#Generating some data with normal distribution at 
#(0,0)
#(8,8)
#(12,0)
#(15,15)
#Data=GenerateData()
#
##Adding some noise with uniform distribution 
##X between [-3,17],
##Y between [-3,17]
#noise=scipy.rand(50,2)*20 -3
#
#Noisy_Data=numpy.concatenate((Data,noise))
#size=20
#
#
#fig = plt.figure()
#ax1=fig.add_subplot(2,1,1) #row, column, figure number
#ax2 = fig.add_subplot(212)
#
#ax1.scatter(Data[:,0],Data[:,1], alpha =  0.5 )
#ax1.scatter(noise[:,0],noise[:,1],color='red' ,alpha =  0.5)
#ax2.scatter(noise[:,0],noise[:,1],color='red' ,alpha =  0.5)

#Epsilon=.2
#MinumumPoints=20
#samples=500
#
#X, clusters = make_circles(n_samples=samples, noise=.05, factor=.5, random_state=0)
##X, clusters = make_blobs(n_samples=samples)
##X, clusters = make_moons(n_samples=samples)
#fig = plt.figure(figsize=[6, 6])
#plt.scatter(X[:,0], X[:,1], c=clusters, cmap='rainbow', alpha=0.7, edgecolors='b')
#s1, s2=numpy.shape(X);
#
#
#result =DBSCAN(X,Epsilon,MinumumPoints)
#
##printed numbers are cluster numbers
##print(result)
##print "Noisy_Data"
##print Noisy_Data.shape
##print Noisy_Data
#
#fig = plt.figure(figsize=[6, 6])
#
#plt.scatter(X[:,0], X[:,1], c=result, cmap='rainbow', alpha=0.7, edgecolors='b')


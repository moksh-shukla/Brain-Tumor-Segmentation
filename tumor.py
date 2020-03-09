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

img_filter1 = Image.open("Y253.jpg").convert('L')
arr = np.asarray(img_filter1)

removed_noise = median_filter(arr, 3) 
arr = Image.fromarray(removed_noise)

plt.imshow(arr, cmap='gray', vmin=0, vmax=255)
plt.show()

size_i=np.shape(arr)
data=(np.array(arr)).flatten()
data = np.reshape(data, (-1, 1))

#k-mean
#clst=cluster.KMeans(n_clusters=2, random_state=None).fit(data);
#Y_p=clst.labels_;  #predicted class after k-mean  

#birch
clst=cluster.Birch(threshold=0.5,n_clusters=2).fit(data);
Y_p=clst.labels_;  #predicted class after k-mean

#clst=cluster.OPTICS(min_samples=5,algorithm='auto').fit(data);
#Y_p=clst.labels_;

#DBSCAN
#clst=DBSCAN(eps=3.2, min_samples=390).fit(data);
#Y_p=clst.labels_;  #predicted class after k-mean  


#fuzzy k-mean
#clst=FuzzyKMeans(2, m=3, max_iter=1000, random_state=0, tol=1e-10).fit(data);
#Y_p=clst.labels_;  #predicted class after k-mean  

#spectral
#clst = cluster.SpectralClustering(n_clusters=2,eigen_solver=None,affinity='nearest_neighbors',n_neighbors=5).fit(data);
#Y_p=clst.labels_;




Y_p=np.resize(Y_p, np.size(Y_p)).reshape(size_i[0],size_i[1])
plt.imshow(Y_p, cmap='gray', vmin=0, vmax=1)
plt.show()






#k-mean with 3 clusters
#for i in range(254):
#clst=cluster.KMeans(n_clusters=2, random_state=None).fit(Y);
#Y_p=clst.labels_;  #predicted class after k-mean








    



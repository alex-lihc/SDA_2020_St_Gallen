
"""
Created on Thu Nov 26 15:31:40 2020
"""
#%%
import os
os.getcwd()
os.chdir( ) # please set your own owrking path
os.getcwd()

#%%
import pandas as pd
import numpy as np 
from scipy import ndimage 
from scipy.cluster import hierarchy 
from scipy.spatial import distance_matrix 
from matplotlib import pyplot as plt 
from sklearn import manifold, datasets 
from sklearn.cluster import AgglomerativeClustering 
from sklearn.datasets.samples_generator import make_blobs
#pip install yellowbrick
import yellowbrick


#Dataset
pdf=pd.read_csv('sp500_dataset.csv')
industry= pdf.groupby(['sector'])['annual_return_log', 'Std' ].mean()
industry = industry.reset_index()

featureset = industry[['annual_return_log', 'Std']]
feature_mtx = featureset.values 

feature500=pdf[['annual_return_log','Std']]
feature500_mtx=feature500.values


#%%
######### Part 1: Hierarchical Clustering

######## Plot1
import scipy
leng = feature_mtx.shape[0]

D = scipy.zeros([leng,leng])
for i in range(leng):
    for j in range(leng):
        D[i,j] = scipy.spatial.distance.euclidean(feature_mtx[i], feature_mtx[j])
import pylab
import scipy.cluster.hierarchy
Z = hierarchy.linkage(D, 'complete')

from scipy.cluster.hierarchy import fcluster
max_d = 5
clusters = fcluster(Z, max_d, criterion='distance')
clusters

#plot the denrogram
fig = pylab.figure(figsize=(9, 12))
def llf(id):
    return '[%s]' % ( (str(industry['sector'][id])) )
dendro = hierarchy.dendrogram(Z,  leaf_label_func=llf, leaf_rotation=0, leaf_font_size =12, orientation = 'right')

plt.savefig('plot 1.png', bbox_inches = 'tight')
plt.show()


#%%
######## Part 2.1: agglomerative clustering
dist_matrix = distance_matrix(feature500_mtx,feature500_mtx) 
agglom = AgglomerativeClustering(n_clusters = 4, linkage = 'complete')
agglom.fit(feature500_mtx)
agglom.labels_
pdf['cluster_'] = agglom.labels_ 
pdf.head()

import matplotlib.cm as cm
n_clusters = max(agglom.labels_)+1
colors = cm.rainbow(np.linspace(0, 1, n_clusters))
cluster_labels = list(range(0, n_clusters))
plt.figure(figsize=(40,45))

for color, label in zip(colors, cluster_labels):
    subset = pdf[pdf.cluster_ == label]
    for i in subset.index:
            plt.text(subset.Std[i], subset.annual_return_log[i], str(subset['Company'][i]), rotation=25) 
    plt.scatter(subset.Std, subset.annual_return_log, s= 50, c=color, label='cluster'+str(label),alpha=0.5)

plt.legend()
plt.title('stock return')
plt.xlabel('standard deviation')
plt.ylabel('annual return')


#%%
####### Part 2.2: agglomerative clustering by industry

df2= pdf.groupby(['cluster_', 'sector'])['annual_return_log', 'Std' ].mean()


plt.figure(figsize=(20,12))
for color, label in zip(colors, cluster_labels):
    subset = df2.loc[(label,),]
    for i in subset.index:
        plt.text(subset.loc[i][1], subset.loc[i][0], str(i), rotation=20)
    plt.scatter(subset.Std, subset.annual_return_log, s=20, c=color, label='cluster'+str(label))
plt.legend()
plt.title('stock return')
plt.xlabel('standard deviation')
plt.ylabel('annual return')




#%%
##### Part 3: K-mean set-up
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans 
from mpl_toolkits.mplot3d import Axes3D 
feature=pdf[['annual_return_log','Skewness','Kurtosis']]
feature_mtx2=feature.values

#%%
#Normalisation
from sklearn.preprocessing import StandardScaler
feature_mtx2 = StandardScaler().fit_transform(feature_mtx2)

#%%
##### Part 3.1: K-mean clustering 3D
clusterNum = 8
k_means = KMeans(init = "k-means++", n_clusters = clusterNum, n_init = 12)
k_means.fit(feature_mtx2)
labels = k_means.labels_

fig = plt.figure(1, figsize=(8, 6))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=20, azim=160)
plt.cla()
ax.set_xlabel('Kurtosis')
ax.set_ylabel('Skewness')
ax.set_zlabel('Annual_return')

ax.scatter(feature_mtx2[:, 2], feature_mtx2[:, 1], feature_mtx2[:, 0], c= labels.astype(np.float))


#%%
##### Part 3.2: K-mean clustering 3D

fig = plt.figure(1, figsize=(8, 6))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=20, azim=100)
plt.cla()
ax.set_xlabel('Kurtosis')
ax.set_ylabel('Skewness')
ax.set_zlabel('Annual_return')

ax.scatter(feature_mtx2[:, 2], feature_mtx2[:, 1], feature_mtx2[:, 0], c= labels.astype(np.float))


#%% choosing the optimal k (This part needs to be done separately, because it eliminates the cluster colors)
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
model = KMeans()
visualizer = KElbowVisualizer(model, k=(4,12))

visualizer.fit(feature_mtx)        # Fit the data to the visualizer
visualizer.show()













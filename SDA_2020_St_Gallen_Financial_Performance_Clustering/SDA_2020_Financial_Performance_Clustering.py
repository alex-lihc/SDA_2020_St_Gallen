#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#%%
import bs4 as bs
import datetime as dt
import os
import pandas as pd
import pandas_datareader.data as web
import pickle
import requests

import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import pandas_datareader as dr
from pandas_datareader import data
from datetime import datetime
import cvxopt as opt
from cvxopt import blas, solvers

from scipy import ndimage 
from scipy.stats import kurtosis, skew
from scipy.cluster import hierarchy 
from scipy.spatial import distance_matrix 
from sklearn import manifold, datasets 
from sklearn.cluster import AgglomerativeClustering 
from sklearn.datasets.samples_generator import make_blobs
#pip install yellowbrick
import yellowbrick


#%%
# set working path
os.getcwd()
os.chdir('C:\Temp\SDA\SDA_2020_St_Gallen\SDA_2020_St_Gallen\SDA_2020_St_Gallen_Clustering_Financial_Performance') # please set your own working path
os.getcwd()

#%%
# set working path

# define object that can collect component stocks of S&P500
def save_sp500_tickers():
   resp = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
   soup = bs.BeautifulSoup(resp.text, 'lxml')
   table = soup.find('table', {'class': 'wikitable sortable'})
   tickers = []
   sectors = []
   for row in table.findAll('tr')[1:]:
       ticker = row.findAll('td')[0].text
       ticker = ticker[:-1]
       if "." in ticker:
            ticker = ticker.replace('.','-')
            print('ticker replaced to', ticker) 
       
       tickers.append(ticker)  
       sector = row.findAll('td')[3].text
       if "\n" in sector:
           sector = sector.replace('\n','')
           print('sector replaced to', sector)
       sectors.append(sector)  
   with open("sp500tickers.pickle","wb") as f:
       pickle.dump(tickers,f)
       
   return tickers, sectors
#%%
tickers,sectors = save_sp500_tickers()

# define object that can collect data from yahoo finance
def get_data_from_yahoo(reload_sp500=False):
    if reload_sp500:
        tickers = save_sp500_tickers()
    else:
        with open("sp500tickers.pickle", "rb") as f:
            tickers = pickle.load(f)
    if not os.path.exists('stock_dfs'):
        os.makedirs('stock_dfs')

    start = datetime(2020, 1, 20)
    end = datetime(2020, 12, 21)
    
    for ticker in tickers:
        # just in case your connection breaks, we'd like to save our progress!
        if not os.path.exists('stock_dfs/{}.csv'.format(ticker)):
            df = web.DataReader(ticker, 'yahoo', start, end)
            df.reset_index(inplace=True)
            df.set_index("Date", inplace=True)
            #df = df.drop("Symbol", axis=1)
            df.to_csv('stock_dfs/{}.csv'.format(ticker))
        else:
            print('Already have {}'.format(ticker))


get_data_from_yahoo()
#%%
# Compile data into one sheet
def compile_data():
    with open("sp500tickers.pickle", "rb") as f:
        tickers = pickle.load(f)

    main_df = pd.DataFrame()

    for count, ticker in enumerate(tickers):
        df = pd.read_csv('stock_dfs/{}.csv'.format(ticker))
        df.set_index('Date', inplace=True)

        df.rename(columns={'Adj Close': ticker}, inplace=True)
        df.drop(['Open', 'High', 'Low', 'Close', 'Volume'], 1, inplace=True)

        if main_df.empty:
            main_df = df
        else:
            main_df = main_df.join(df, how='outer')

        if count % 10 == 0:
            print(count)
    print(main_df.head())
    main_df.to_csv('sp500_joined_closes.csv')


compile_data()

#%%
# Compute returns and drop the columns with zero values
df = pd.read_csv('sp500_joined_closes.csv')
df.set_index('Date', inplace = True)


#%%
#look if there are values missing
print(df.isnull().sum())

#drop the stock if more than 5 prices are missing, otherwise replace the missing values with the values in the previous row
df=df.dropna(axis=1,thresh=5)
df=df.fillna(axis=1, method='ffill')
print(df.isnull().sum())

df
prices = df
prices

#%%
# Calculate the log returns
log_r = np.log(prices / prices.shift(1))
log_r = log_r.drop(axis = 0, index = ['2020-01-21'])
log_r 

# Compute the annualised returns
annual_r = log_r.mean() * 252
annual_r.name = 'annual return log'
annual_r
 
# Calculate the covariance matrix
cov_matrix = log_r.cov() * 252
cov_matrix

# Calculate the volatility
var = log_r.var() * 252
Std = np.sqrt(var)
Std.name = 'Std'

#%%

# Calculate Skewness and Kurtosis
index=annual_r.index
Skewness=pd.DataFrame(skew(log_r))
Kurtosis=pd.DataFrame(kurtosis(log_r))

Skewness.set_index(index, inplace=True)
Kurtosis.set_index(index, inplace=True)

Skewness.columns=['Skewness']
Kurtosis.columns=['Kurtosis']




#%%
# Compile the dataset we need

pd_sectors = pd.Series(sectors, index = tickers)
pd_sectors.name = 'sector'

dataset = pd.concat([pd_sectors, annual_r, Std, Skewness, Kurtosis], axis = 1)
dataset.drop(['CARR','LUMN' ,'OTIS' ,'VNT','TSLA', 'VIAC'], inplace=True)
dataset.reset_index(level=0, inplace=True)
dataset.columns=['Company', 'sector', 'annual_return_log', 'Std','Skewness','Kurtosis']

# save the dataset
dataset.to_csv('sp500_dataset.csv')


#%%

# Prepare Dataset
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






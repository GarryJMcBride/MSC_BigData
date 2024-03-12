import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import metrics
from sklearn import cluster
from sklearn.preprocessing import scale
import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering

premierleagueplayers = pd.read_csv("Statistical Analysis of Fantasy Premier League\Data\FPL_2018_19_Wk7.csv")


premierleagueplayers = premierleagueplayers.drop(columns=['Name', 'Team', 'Position'])
print(premierleagueplayers)

premierleagueplayers.head()
# print(premierleagueplayers.head())



X = premierleagueplayers.values[:, 8:14]
Y = premierleagueplayers.values[:, 7]
# print(X)
# print(Y)

scaled_premierleagueplayers = scale(X)
# print(scaled_premierleagueplayers)


# this is  Hierarchical Clustering
# # try to get socre between 0-1,1 closer is better
from sklearn import cluster
from sklearn.preprocessing import LabelEncoder
n_samples, n_features = scaled_premierleagueplayers.shape
n_digits = len(np.unique(Y))
Y2 = LabelEncoder().fit_transform(Y)
# go online to find possible arguments for average and cosine
model = cluster.AgglomerativeClustering(n_clusters=n_digits, linkage="average", affinity="cosine")
model.fit(scaled_premierleagueplayers)
print(Y2)
print(model.labels_)
print(metrics.silhouette_score(scaled_premierleagueplayers, model.labels_))
print(metrics.completeness_score(Y2, model.labels_))
print(metrics.homogeneity_score(Y2, model.labels_))


#this is general clustering
from sklearn import cluster
from sklearn.preprocessing import LabelEncoder

n_samples, n_features = scaled_premierleagueplayers.shape
n_digits = len(np.unique(Y))
Y2 = LabelEncoder().fit_transform(Y)
aff = ["euclidean", "l1", "l2", "manhattan", "cosine"]
link = ["ward", "complete", "average"]
for a in aff:
    for l in link:
        if (l == "ward" and a != "euclidean"):
            continue
        else:
            print(a, l)
            model = cluster.AgglomerativeClustering(n_clusters=n_digits, linkage=l, affinity=a)
            model.fit(scaled_premierleagueplayers)
            print(metrics.silhouette_score(scaled_premierleagueplayers, model.labels_))
            print(metrics.completeness_score(Y2, model.labels_))
            print(metrics.homogeneity_score(Y2, model.labels_))


#this is k means clustering, change range to get better score
from sklearn import cluster
from sklearn.preprocessing import LabelEncoder

n_samples, n_features = scaled_premierleagueplayers.shape
n_digits = len(np.unique(Y))
Y2 = LabelEncoder().fit_transform(Y)
for k in range(2, 20):
    kmeans = cluster.KMeans(n_clusters=k)
    kmeans.fit(scaled_premierleagueplayers)
    print(k)
    print(metrics.silhouette_score(scaled_premierleagueplayers, kmeans.labels_))
    print(metrics.completeness_score(Y2, kmeans.labels_))
    print(metrics.homogeneity_score(Y2, kmeans.labels_))

#denogram
from scipy.cluster.hierarchy import dendrogram, linkage

model = linkage(premierleagueplayers, 'ward')
Y2 = LabelEncoder().fit_transform(Y)
plt.figure()
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('sample index')
plt.ylabel('distance')
dendrogram(model, leaf_rotation=90., leaf_font_size=8.,)
plt.show()